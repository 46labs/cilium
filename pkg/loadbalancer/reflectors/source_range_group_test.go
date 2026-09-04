// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package reflectors

import (
	"net/netip"
	"testing"

	"github.com/cilium/hive/hivetest"
	"github.com/cilium/statedb"
	"github.com/stretchr/testify/require"

	cmtypes "github.com/cilium/cilium/pkg/clustermesh/types"
	"github.com/cilium/cilium/pkg/loadbalancer"
)

// TestUpdateBackendsWithSourceRangeGroup covers the join between the
// lb-source-range-group-pods table and a Backend's SourceRanges field --
// this is the code path convertEndpoints invokes (via the transformBackend
// callback) for every backend derived from a real EndpointSlice, and is what
// makes the per-pod source/port range restriction take effect end-to-end.
func TestUpdateBackendsWithSourceRangeGroup(t *testing.T) {
	db := statedb.New()
	pods, err := NewPodTable(db)
	require.NoError(t, err, "NewPodTable")

	wtxn := db.WriteTxn(pods)
	_, _, err = pods.Insert(wtxn, LbSrcRangeGroupPod{
		UID:          "pod-uid",
		Namespace:    "test-ns",
		Name:         "pinned-pod",
		IP:           netip.MustParseAddr("10.1.0.1"),
		SourceRanges: "10.0.0.0/8,10.1.0.0/16:5060",
	})
	require.NoError(t, err, "Insert")
	wtxn.Commit()

	p := reflectorParams{DB: db, Log: hivetest.Logger(t), LbSrcRangeGroupPods: pods}

	newBackend := func(addr string) *loadbalancer.Backend {
		return &loadbalancer.Backend{
			Address: loadbalancer.NewL3n4Addr(
				loadbalancer.TCP,
				cmtypes.MustParseAddrCluster(addr),
				80,
				loadbalancer.ScopeExternal,
			),
		}
	}

	t.Run("backend whose IP matches a pinned pod gets SourceRanges set", func(t *testing.T) {
		be := newBackend("10.1.0.1")
		p.updateBackendsWithSourceRangeGroup(be)
		require.Equal(t, []loadbalancer.SourceAndPortRangeEntry{
			{Prefix: netip.MustParsePrefix("10.0.0.0/8")},
			{Prefix: netip.MustParsePrefix("10.1.0.0/16"), Port: 5060},
		}, be.SourceRanges)
	})

	t.Run("backend with no matching pinned pod is left untouched", func(t *testing.T) {
		be := newBackend("10.1.0.2")
		p.updateBackendsWithSourceRangeGroup(be)
		require.Nil(t, be.SourceRanges, "unpinned backend must not get SourceRanges")
	})

	t.Run("pinned pod with unparseable source ranges leaves the backend untouched", func(t *testing.T) {
		wtxn := db.WriteTxn(pods)
		_, _, err := pods.Insert(wtxn, LbSrcRangeGroupPod{
			UID:          "bad-pod-uid",
			Namespace:    "test-ns",
			Name:         "bad-pod",
			IP:           netip.MustParseAddr("10.1.0.3"),
			SourceRanges: "not-a-cidr",
		})
		require.NoError(t, err, "Insert")
		wtxn.Commit()

		be := newBackend("10.1.0.3")
		p.updateBackendsWithSourceRangeGroup(be)
		require.Nil(t, be.SourceRanges, "backend must not get SourceRanges from an unparseable pod entry")
	})

	t.Run("pinned pod with the annotation removed clears any previously-set SourceRanges", func(t *testing.T) {
		// Re-insert the same pod (same Namespace/Name primary key) with an
		// empty SourceRanges, mirroring what the reflector now reports when
		// the pod's PodSourceRanges annotation is removed but the pod (and
		// its selecting label) still exist.
		wtxn := db.WriteTxn(pods)
		_, _, err := pods.Insert(wtxn, LbSrcRangeGroupPod{
			UID:       "pod-uid",
			Namespace: "test-ns",
			Name:      "pinned-pod",
			IP:        netip.MustParseAddr("10.1.0.1"),
		})
		require.NoError(t, err, "Insert")
		wtxn.Commit()

		be := newBackend("10.1.0.1")
		be.SourceRanges = []loadbalancer.SourceAndPortRangeEntry{{Prefix: netip.MustParsePrefix("10.0.0.0/8")}}
		p.updateBackendsWithSourceRangeGroup(be)
		require.Empty(t, be.SourceRanges, "stale SourceRanges must be cleared when the pod's annotation is removed")
	})

	t.Run("pinned pod removed: the backend is no longer resolved", func(t *testing.T) {
		wtxn := db.WriteTxn(pods)
		require.NoError(t, pods.DeleteAll(wtxn), "DeleteAll")
		wtxn.Commit()

		be := newBackend("10.1.0.1")
		p.updateBackendsWithSourceRangeGroup(be)
		require.Nil(t, be.SourceRanges, "removed pin must no longer resolve")
	})
}
