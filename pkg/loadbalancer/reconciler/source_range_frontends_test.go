// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package reconciler

import (
	"net/netip"
	"slices"
	"testing"

	"github.com/cilium/hive/hivetest"
	"github.com/cilium/statedb"
	"github.com/stretchr/testify/require"

	cmtypes "github.com/cilium/cilium/pkg/clustermesh/types"
	"github.com/cilium/cilium/pkg/loadbalancer"
	"github.com/cilium/cilium/pkg/loadbalancer/reflectors"
	"github.com/cilium/cilium/pkg/source"
)

// TestApplySourceRangesForFrontend covers the start-lb-source-range-frontends-observer
// job's core logic: re-deriving Backend.SourceRanges from the current
// LbSrcRangeGroupPods table whenever a frontend appears or changes.
func TestApplySourceRangesForFrontend(t *testing.T) {
	p := newApplySourceRangePodFixture(t)
	log := hivetest.Logger(t)

	pods, err := reflectors.NewPodTable(p.DB)
	require.NoError(t, err, "NewPodTable")

	ops := &BPFOps{
		db:                  p.DB,
		fes:                 p.Frontends,
		writer:              p.Writer,
		log:                 newRateLimitingLogger(log),
		lbSrcRangeGroupPods: pods,
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

	// setup (re-)creates the service/frontend/backend from scratch, so each
	// subtest starts from a backend with no SourceRanges regardless of what
	// earlier subtests left behind.
	setup := func(t *testing.T, enabled bool) {
		t.Helper()
		txn := p.Writer.WriteTxn()
		require.NoError(t, p.Writer.UpsertServiceAndFrontends(
			txn,
			&loadbalancer.Service{
				Name:                        svcName,
				Source:                      source.Kubernetes,
				SourceAndPortRangeLbEnabled: enabled,
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
	}

	getFrontend := func(t *testing.T) *loadbalancer.Frontend {
		t.Helper()
		fe, _, found := p.Frontends.Get(p.DB.ReadTxn(), loadbalancer.FrontendByAddress(feAddr))
		require.True(t, found, "frontend must exist")
		return fe
	}

	getBackend := func(t *testing.T) (*loadbalancer.Backend, statedb.Revision) {
		t.Helper()
		be, rev, found := p.Backends.Get(p.DB.ReadTxn(), loadbalancer.BackendByAddress(beAddr))
		require.True(t, found, "backend must exist")
		return be, rev
	}

	setPod := func(t *testing.T, sourceRanges string) {
		t.Helper()
		wtxn := p.DB.WriteTxn(pods)
		_, _, err := pods.Insert(wtxn, reflectors.LbSrcRangeGroupPod{
			UID:          "pod-uid",
			Namespace:    "test",
			Name:         "pod",
			IP:           beAddr.AddrCluster().Addr(),
			SourceRanges: sourceRanges,
		})
		require.NoError(t, err, "Insert")
		wtxn.Commit()
	}

	clearPod := func(t *testing.T) {
		t.Helper()
		wtxn := p.DB.WriteTxn(pods)
		require.NoError(t, pods.DeleteAll(wtxn), "DeleteAll")
		wtxn.Commit()
	}

	t.Run("backend created before the pod's source ranges were reflected gets backfilled once the frontend refreshes", func(t *testing.T) {
		setup(t, true)
		t.Cleanup(func() { clearPod(t) })

		// Simulate the race: the backend already exists (no SourceRanges, as
		// if endpoint conversion ran before the pod's LbSrcRangeGroupPods row
		// existed), and only now does the pod row appear.
		be, _ := getBackend(t)
		require.Empty(t, be.SourceRanges, "precondition: backend must start without SourceRanges")

		setPod(t, "10.0.0.0/8,10.1.0.0/16:5060")

		ops.applySourceRangesForFrontend(p.DB.ReadTxn(), getFrontend(t))

		be, _ = getBackend(t)
		require.Equal(t, []loadbalancer.SourceAndPortRangeEntry{
			{Prefix: netip.MustParsePrefix("10.0.0.0/8")},
			{Prefix: netip.MustParsePrefix("10.1.0.0/16"), Port: 5060},
		}, be.SourceRanges)
	})

	t.Run("frontend with the feature disabled is left untouched even with a matching pod", func(t *testing.T) {
		setup(t, false)
		t.Cleanup(func() { clearPod(t) })

		setPod(t, "10.0.0.0/8")

		ops.applySourceRangesForFrontend(p.DB.ReadTxn(), getFrontend(t))

		be, _ := getBackend(t)
		require.Empty(t, be.SourceRanges, "SourceRanges must not be applied when the service has the feature disabled")
	})

	t.Run("pod removed clears a previously-applied SourceRanges once the frontend refreshes", func(t *testing.T) {
		setup(t, true)
		setPod(t, "10.0.0.0/8")
		ops.applySourceRangesForFrontend(p.DB.ReadTxn(), getFrontend(t))
		be, _ := getBackend(t)
		require.NotEmpty(t, be.SourceRanges, "precondition: backend must have SourceRanges set")

		clearPod(t)

		ops.applySourceRangesForFrontend(p.DB.ReadTxn(), getFrontend(t))
		be, _ = getBackend(t)
		require.Empty(t, be.SourceRanges, "SourceRanges must be cleared once the pinning pod is gone")
	})

	t.Run("re-applying with an already up-to-date frontend is a no-op", func(t *testing.T) {
		// Guards against the frontends-observer job looping forever: its own
		// UpsertBackends call refreshes the frontend, which feeds right back
		// into the observer. A second call with that refreshed frontend must
		// not write again.
		setup(t, true)
		t.Cleanup(func() { clearPod(t) })
		setPod(t, "10.0.0.0/8")

		ops.applySourceRangesForFrontend(p.DB.ReadTxn(), getFrontend(t))
		_, revAfterFirst := getBackend(t)

		// Second call, as the observer job would receive from the very Fes
		// change its own first call caused.
		ops.applySourceRangesForFrontend(p.DB.ReadTxn(), getFrontend(t))
		_, revAfterSecond := getBackend(t)

		require.Equal(t, revAfterFirst, revAfterSecond, "no redundant write should occur once SourceRanges already match the pod table")
	})
}
