// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package reconciler

import (
	"context"
	"log/slog"
	"net/netip"
	"slices"
	"testing"

	"github.com/cilium/hive/cell"
	"github.com/cilium/hive/hivetest"
	"github.com/cilium/statedb"
	"github.com/stretchr/testify/require"

	cmtypes "github.com/cilium/cilium/pkg/clustermesh/types"
	"github.com/cilium/cilium/pkg/datapath/tables"
	"github.com/cilium/cilium/pkg/hive"
	"github.com/cilium/cilium/pkg/kpr"
	"github.com/cilium/cilium/pkg/loadbalancer"
	"github.com/cilium/cilium/pkg/loadbalancer/reflectors"
	"github.com/cilium/cilium/pkg/loadbalancer/writer"
	"github.com/cilium/cilium/pkg/node"
	"github.com/cilium/cilium/pkg/option"
	"github.com/cilium/cilium/pkg/source"
)

// applySourceRangePodTestParams mirrors writer.testParams (writer/writer_test.go)
// so we can obtain a real *writer.Writer and its backing tables without
// reimplementing the writer module's wiring.
type applySourceRangePodTestParams struct {
	cell.In

	DB        *statedb.DB
	Writer    *writer.Writer
	Frontends statedb.Table[*loadbalancer.Frontend]
	Backends  statedb.Table[*loadbalancer.Backend]
}

func newApplySourceRangePodFixture(t *testing.T) applySourceRangePodTestParams {
	t.Helper()
	log := hivetest.Logger(t, hivetest.LogLevel(slog.LevelError))

	var p applySourceRangePodTestParams
	h := hive.New(
		loadbalancer.ConfigCell,
		node.LocalNodeStoreTestCell,
		writer.Cell,
		cell.Provide(
			func() cmtypes.ClusterInfo { return cmtypes.ClusterInfo{} },
			func() *option.DaemonConfig { return &option.DaemonConfig{} },
			tables.NewNodeAddressTable,
			statedb.RWTable[tables.NodeAddress].ToTable,
			source.NewSources,
			func() kpr.KPRConfig { return kpr.KPRConfig{} },
		),
		cell.Invoke(func(p_ applySourceRangePodTestParams) { p = p_ }),
	)
	require.NoError(t, h.Start(log, context.Background()), "Start")
	t.Cleanup(func() {
		require.NoError(t, h.Stop(log, context.Background()), "Stop")
	})
	return p
}

// TestApplySourceRangePod covers the start-lb-source-range-pods-observer job's
// core logic: resolving a [reflectors.LbSrcRangeGroupPod] update or deletion
// to the matching backend(s) and applying (or clearing) their SourceRanges.
func TestApplySourceRangePod(t *testing.T) {
	p := newApplySourceRangePodFixture(t)
	log := hivetest.Logger(t)

	ops := &BPFOps{
		db:     p.DB,
		fes:    p.Frontends,
		writer: p.Writer,
		log:    newRateLimitingLogger(log),
	}

	svcName := loadbalancer.NewServiceName("test", "svc")
	beAddr := loadbalancer.NewL3n4Addr(
		loadbalancer.TCP,
		cmtypes.MustParseAddrCluster("10.1.0.1"),
		80,
		loadbalancer.ScopeExternal,
	)
	feAddr := loadbalancer.NewL3n4Addr(
		loadbalancer.TCP,
		cmtypes.MustParseAddrCluster("10.0.0.1"),
		80,
		loadbalancer.ScopeExternal,
	)

	txn := p.Writer.WriteTxn()
	require.NoError(t, p.Writer.UpsertServiceAndFrontends(
		txn,
		&loadbalancer.Service{
			Name:                        svcName,
			Source:                      source.Kubernetes,
			SourceAndPortRangeLbEnabled: true,
		},
		loadbalancer.FrontendParams{
			Type:        loadbalancer.SVCTypeClusterIP,
			ServiceName: svcName,
			Address:     feAddr,
		},
	), "UpsertServiceAndFrontends")
	require.NoError(t, p.Writer.UpsertBackends(
		txn, svcName, source.Kubernetes,
		slices.Values([]loadbalancer.Backend{{
			Address: beAddr,
			State:   loadbalancer.BackendStateActive,
			Weight:  loadbalancer.DefaultBackendWeight,
		}}),
	), "UpsertBackends")
	txn.Commit()

	getBackendSourceRanges := func() []loadbalancer.SourceAndPortRangeEntry {
		t.Helper()
		be, _, found := p.Backends.Get(p.DB.ReadTxn(), loadbalancer.BackendByAddress(beAddr))
		require.True(t, found, "backend must exist")
		return be.SourceRanges
	}

	pod := reflectors.LbSrcRangeGroupPod{
		UID:          "pod-uid",
		Namespace:    "test",
		Name:         "pod",
		IP:           beAddr.AddrCluster().Addr(),
		SourceRanges: "10.0.0.0/8,10.1.0.0/16:5060",
	}

	t.Run("pod update sets SourceRanges on the matching backend", func(t *testing.T) {
		ops.applySourceRangePod(p.DB.ReadTxn(), pod, false)
		require.Equal(t, []loadbalancer.SourceAndPortRangeEntry{
			{Prefix: netip.MustParsePrefix("10.0.0.0/8")},
			{Prefix: netip.MustParsePrefix("10.1.0.0/16"), Port: 5060},
		}, getBackendSourceRanges())
	})

	t.Run("pod deletion clears SourceRanges on the matching backend", func(t *testing.T) {
		ops.applySourceRangePod(p.DB.ReadTxn(), pod, true)
		require.Empty(t, getBackendSourceRanges(), "SourceRanges must be cleared on deletion")
	})

	t.Run("pod update with empty source ranges (annotation removed) clears SourceRanges without erroring", func(t *testing.T) {
		// Re-apply real ranges first so this sub-test can observe them being cleared.
		ops.applySourceRangePod(p.DB.ReadTxn(), pod, false)
		require.NotEmpty(t, getBackendSourceRanges(), "precondition: backend must have SourceRanges set")

		clearedPod := pod
		clearedPod.SourceRanges = ""
		ops.applySourceRangePod(p.DB.ReadTxn(), clearedPod, false)
		require.Empty(t, getBackendSourceRanges(), "SourceRanges must be cleared when the pod annotation is removed")
	})

	t.Run("pod with an unparseable source ranges value leaves the backend untouched", func(t *testing.T) {
		ops.applySourceRangePod(p.DB.ReadTxn(), pod, false)
		require.NotEmpty(t, getBackendSourceRanges(), "precondition: backend must have SourceRanges set")

		badPod := pod
		badPod.SourceRanges = "not-a-cidr"
		ops.applySourceRangePod(p.DB.ReadTxn(), badPod, false)
		require.NotEmpty(t, getBackendSourceRanges(), "backend's SourceRanges must be left alone on a parse error")
	})

	t.Run("pod with no matching backend is a no-op", func(t *testing.T) {
		unrelated := pod
		unrelated.IP = netip.MustParseAddr("10.9.9.9")
		require.NotPanics(t, func() {
			ops.applySourceRangePod(p.DB.ReadTxn(), unrelated, false)
		})
	})
}
