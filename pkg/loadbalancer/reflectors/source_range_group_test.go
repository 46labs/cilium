// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package reflectors

import (
	"net/netip"
	"testing"

	"github.com/cilium/statedb"
	"github.com/stretchr/testify/require"

	cmtypes "github.com/cilium/cilium/pkg/clustermesh/types"
	"github.com/cilium/cilium/pkg/loadbalancer"
)

// TestUpdateBackendsWithSourceRangeGroup covers the join between the
// lb-source-range-group-pods table and a Backend's SourceRangeGroup field --
// this is the code path convertEndpoints invokes (via the transformBackend
// callback) for every backend derived from a real EndpointSlice, and is what
// makes ServiceSourceRangeIndex pinning actually take effect end-to-end.
func TestUpdateBackendsWithSourceRangeGroup(t *testing.T) {
	db := statedb.New()
	pods, err := NewPodTable(db)
	require.NoError(t, err, "NewPodTable")

	wtxn := db.WriteTxn(pods)
	_, _, err = pods.Insert(wtxn, LbSrcRangeGroupPod{
		UID:        "pod-uid",
		Namespace:  "test-ns",
		Name:       "pinned-pod",
		IP:         netip.MustParseAddr("10.1.0.1"),
		GroupIndex: 3,
	})
	require.NoError(t, err, "Insert")
	wtxn.Commit()

	p := reflectorParams{DB: db, LbSrcRangeGroupPods: pods}

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

	t.Run("backend whose IP matches a pinned pod gets SourceRangeGroup set", func(t *testing.T) {
		be := newBackend("10.1.0.1")
		p.updateBackendsWithSourceRangeGroup(be)
		require.NotNil(t, be.SourceRangeGroup, "expected SourceRangeGroup to be resolved")
		require.Equal(t, uint8(3), *be.SourceRangeGroup)
	})

	t.Run("backend with no matching pinned pod is left untouched", func(t *testing.T) {
		be := newBackend("10.1.0.2")
		p.updateBackendsWithSourceRangeGroup(be)
		require.Nil(t, be.SourceRangeGroup, "unpinned backend must not get a SourceRangeGroup")
	})

	t.Run("pinned pod removed: the backend is no longer resolved", func(t *testing.T) {
		wtxn := db.WriteTxn(pods)
		require.NoError(t, pods.DeleteAll(wtxn), "DeleteAll")
		wtxn.Commit()

		be := newBackend("10.1.0.1")
		p.updateBackendsWithSourceRangeGroup(be)
		require.Nil(t, be.SourceRangeGroup, "removed pin must no longer resolve")
	})
}
