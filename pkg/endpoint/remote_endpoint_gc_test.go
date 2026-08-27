// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package endpoint

import (
	"context"
	"net"
	"net/netip"
	"testing"

	"github.com/cilium/hive/hivetest"
	"github.com/cilium/stream"
	"github.com/stretchr/testify/require"
	k8sRuntime "k8s.io/apimachinery/pkg/runtime"

	"github.com/cilium/cilium/pkg/datapath/tunnel"
	v2 "github.com/cilium/cilium/pkg/k8s/apis/cilium.io/v2"
	cilium_v2a1 "github.com/cilium/cilium/pkg/k8s/apis/cilium.io/v2alpha1"
	"github.com/cilium/cilium/pkg/k8s/resource"
	"github.com/cilium/cilium/pkg/k8s/types"
	"github.com/cilium/cilium/pkg/lock"
	"github.com/cilium/cilium/pkg/maps/ctmap"
	"github.com/cilium/cilium/pkg/node"
	"github.com/cilium/cilium/pkg/node/addressing"
	nodeTypes "github.com/cilium/cilium/pkg/node/types"
	"github.com/cilium/cilium/pkg/option"
	"github.com/cilium/cilium/pkg/time"
	"github.com/cilium/cilium/pkg/trigger"
)

const (
	localNodeIP  = "10.0.0.1"
	remoteNodeIP = "10.0.0.2"
)

// recordingGCRunner is a ctmap.GCRunner that records every filter it is invoked with.
type recordingGCRunner struct {
	mu    lock.Mutex
	calls []ctmap.GCFilter
}

func (g *recordingGCRunner) Run(filter ctmap.GCFilter) (int, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.calls = append(g.calls, filter)
	return len(filter.MatchIPs), nil
}

func (g *recordingGCRunner) Observe4() stream.Observable[ctmap.GCEvent] {
	return stream.FuncObservable[ctmap.GCEvent](func(context.Context, func(ctmap.GCEvent), func(error)) {})
}

func (g *recordingGCRunner) Observe6() stream.Observable[ctmap.GCEvent] {
	return stream.FuncObservable[ctmap.GCEvent](func(context.Context, func(ctmap.GCEvent), func(error)) {})
}

func (g *recordingGCRunner) snapshot() []ctmap.GCFilter {
	g.mu.Lock()
	defer g.mu.Unlock()
	out := make([]ctmap.GCFilter, len(g.calls))
	copy(out, g.calls)
	return out
}

// fakeResource is a minimal resource.Resource[T] whose events are fed manually by the test.
type fakeResource[T k8sRuntime.Object] struct {
	events chan resource.Event[T]
}

func newFakeResource[T k8sRuntime.Object]() *fakeResource[T] {
	return &fakeResource[T]{events: make(chan resource.Event[T])}
}

func (f *fakeResource[T]) Events(_ context.Context, _ ...resource.EventsOpt) <-chan resource.Event[T] {
	return f.events
}

func (f *fakeResource[T]) Store(context.Context) (resource.Store[T], error) {
	return nil, nil
}

func (f *fakeResource[T]) Observe(ctx context.Context, next func(resource.Event[T]), complete func(error)) {
	for ev := range f.events {
		next(ev)
	}
	complete(nil)
}

// send pushes an event and blocks until the consumer marks it Done, mirroring the
// synchronous processing model of resource.Resource.
func send[T k8sRuntime.Object](t *testing.T, f *fakeResource[T], kind resource.EventKind, key resource.Key, obj T) {
	t.Helper()
	done := make(chan error, 1)
	select {
	case f.events <- resource.Event[T]{Kind: kind, Key: key, Object: obj, Done: func(err error) { done <- err }}:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out sending event")
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for event to be marked done")
	}
}

func newTestRemoteEndpointGC(t *testing.T, gc ctmap.GCRunner, cesEnabled bool) *remoteEndpointGC {
	t.Helper()

	old := option.Config.EnableIPv4
	option.Config.EnableIPv4 = true
	t.Cleanup(func() { option.Config.EnableIPv4 = old })

	mockNode := node.LocalNode{
		Node: nodeTypes.Node{
			IPAddresses: []nodeTypes.Address{
				{Type: addressing.NodeInternalIP, IP: net.ParseIP(localNodeIP)},
			},
		},
		Local: &node.LocalNodeInfo{UnderlayProtocol: tunnel.IPv4},
	}

	r := &remoteEndpointGC{
		log:                        hivetest.Logger(t),
		ctMapGC:                    gc,
		localNode:                  node.NewTestLocalNodeStore(mockNode),
		ciliumEndpointSliceEnabled: cesEnabled,
		gcBatch:                    map[ctmap.NetAddr]struct{}{},
	}

	tr, err := trigger.NewTrigger(trigger.Parameters{
		Name: "test-remote-endpoint-ct-gc-batching",
		TriggerFunc: func([]string) {
			r.mu.Lock()
			batch := r.gcBatch
			r.gcBatch = map[ctmap.NetAddr]struct{}{}
			r.mu.Unlock()

			if len(batch) == 0 {
				return
			}
			_, _ = r.ctMapGC.Run(ctmap.GCFilter{MatchIPs: batch})
		},
	})
	require.NoError(t, err)
	r.trigger = tr

	return r
}

func addrOf(t *testing.T, s string) ctmap.NetAddr {
	t.Helper()
	a, err := netip.ParseAddr(s)
	require.NoError(t, err)
	return ctmap.NetAddr{Addr: a}
}

func networking(nodeIP string, ipv4, ipv6 string) *v2.EndpointNetworking {
	pair := &v2.AddressPair{}
	if ipv4 != "" {
		pair.IPV4 = ipv4
	}
	if ipv6 != "" {
		pair.IPV6 = ipv6
	}
	return &v2.EndpointNetworking{
		NodeIP:     nodeIP,
		Addressing: v2.AddressPairList{pair},
	}
}

func TestEndpointAddrs(t *testing.T) {
	r := &remoteEndpointGC{log: hivetest.Logger(t)}

	tests := []struct {
		name     string
		networks []*v2.EndpointNetworking
		want     map[ctmap.NetAddr]struct{}
	}{
		{
			name:     "empty",
			networks: nil,
			want:     map[ctmap.NetAddr]struct{}{},
		},
		{
			name:     "ipv4 only, no error logged for empty ipv6",
			networks: []*v2.EndpointNetworking{networking(remoteNodeIP, "10.1.0.5", "")},
			want:     map[ctmap.NetAddr]struct{}{addrOf(t, "10.1.0.5"): {}},
		},
		{
			name:     "ipv6 only, no error logged for empty ipv4",
			networks: []*v2.EndpointNetworking{networking(remoteNodeIP, "", "fd00::5")},
			want:     map[ctmap.NetAddr]struct{}{addrOf(t, "fd00::5"): {}},
		},
		{
			name:     "dual stack",
			networks: []*v2.EndpointNetworking{networking(remoteNodeIP, "10.1.0.5", "fd00::5")},
			want: map[ctmap.NetAddr]struct{}{
				addrOf(t, "10.1.0.5"): {},
				addrOf(t, "fd00::5"):  {},
			},
		},
		{
			name: "invalid address is skipped, valid ones still collected",
			networks: []*v2.EndpointNetworking{
				networking(remoteNodeIP, "not-an-ip", ""),
				networking(remoteNodeIP, "10.1.0.9", ""),
			},
			want: map[ctmap.NetAddr]struct{}{addrOf(t, "10.1.0.9"): {}},
		},
		{
			name: "multiple endpoints and address pairs",
			networks: []*v2.EndpointNetworking{
				networking(remoteNodeIP, "10.1.0.5", ""),
				networking(remoteNodeIP, "10.1.0.6", ""),
			},
			want: map[ctmap.NetAddr]struct{}{
				addrOf(t, "10.1.0.5"): {},
				addrOf(t, "10.1.0.6"): {},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, r.endpointAddrs(tt.networks))
		})
	}
}

func TestRemovedCESEndpoints(t *testing.T) {
	epA := cilium_v2a1.CoreCiliumEndpoint{Name: "a", Networking: networking(remoteNodeIP, "10.1.0.1", "")}
	epB := cilium_v2a1.CoreCiliumEndpoint{Name: "b", Networking: networking(remoteNodeIP, "10.1.0.2", "")}
	epBUpdated := cilium_v2a1.CoreCiliumEndpoint{Name: "b", Networking: networking(remoteNodeIP, "10.1.0.99", "")}

	t.Run("first observation of a slice removes nothing", func(t *testing.T) {
		newCES := &cilium_v2a1.CiliumEndpointSlice{Endpoints: []cilium_v2a1.CoreCiliumEndpoint{epA, epB}}
		require.Empty(t, removedCESEndpoints(nil, newCES))
	})

	t.Run("endpoint dropped from the slice is reported as removed", func(t *testing.T) {
		oldCES := &cilium_v2a1.CiliumEndpointSlice{Endpoints: []cilium_v2a1.CoreCiliumEndpoint{epA, epB}}
		newCES := &cilium_v2a1.CiliumEndpointSlice{Endpoints: []cilium_v2a1.CoreCiliumEndpoint{epB}}
		require.Equal(t, []cilium_v2a1.CoreCiliumEndpoint{epA}, removedCESEndpoints(oldCES, newCES))
	})

	t.Run("endpoint present in both versions is not reported, even if its contents changed", func(t *testing.T) {
		oldCES := &cilium_v2a1.CiliumEndpointSlice{Endpoints: []cilium_v2a1.CoreCiliumEndpoint{epA, epB}}
		newCES := &cilium_v2a1.CiliumEndpointSlice{Endpoints: []cilium_v2a1.CoreCiliumEndpoint{epA, epBUpdated}}
		require.Empty(t, removedCESEndpoints(oldCES, newCES))
	})

	t.Run("no endpoints removed", func(t *testing.T) {
		oldCES := &cilium_v2a1.CiliumEndpointSlice{Endpoints: []cilium_v2a1.CoreCiliumEndpoint{epA}}
		newCES := &cilium_v2a1.CiliumEndpointSlice{Endpoints: []cilium_v2a1.CoreCiliumEndpoint{epA, epB}}
		require.Empty(t, removedCESEndpoints(oldCES, newCES))
	})
}

func TestRemoteNetworkings(t *testing.T) {
	local := cilium_v2a1.CoreCiliumEndpoint{Name: "local", Networking: networking(localNodeIP, "10.1.0.1", "")}
	remote := cilium_v2a1.CoreCiliumEndpoint{Name: "remote", Networking: networking(remoteNodeIP, "10.1.0.2", "")}
	noNetworking := cilium_v2a1.CoreCiliumEndpoint{Name: "no-net"}

	got := remoteNetworkings([]cilium_v2a1.CoreCiliumEndpoint{local, remote, noNetworking}, localNodeIP)
	require.Equal(t, []*v2.EndpointNetworking{remote.Networking}, got)
}

func TestOnDeleteBatchesAndTriggersGC(t *testing.T) {
	gc := &recordingGCRunner{}
	r := newTestRemoteEndpointGC(t, gc, false)
	defer r.trigger.Shutdown()

	r.onDelete(map[ctmap.NetAddr]struct{}{addrOf(t, "10.1.0.5"): {}})
	r.onDelete(map[ctmap.NetAddr]struct{}{addrOf(t, "10.1.0.6"): {}})

	require.Eventually(t, func() bool { return len(gc.snapshot()) > 0 }, 2*time.Second, 5*time.Millisecond)

	calls := gc.snapshot()
	require.Len(t, calls, 1, "the two onDelete calls should have been folded into a single batched GC run")
	require.Equal(t, map[ctmap.NetAddr]struct{}{
		addrOf(t, "10.1.0.5"): {},
		addrOf(t, "10.1.0.6"): {},
	}, calls[0].MatchIPs)
}

func TestOnDeleteWithNoAddrsDoesNotTriggerGC(t *testing.T) {
	gc := &recordingGCRunner{}
	r := newTestRemoteEndpointGC(t, gc, false)
	defer r.trigger.Shutdown()

	r.onDelete(map[ctmap.NetAddr]struct{}{})

	require.Never(t, func() bool { return len(gc.snapshot()) > 0 }, 200*time.Millisecond, 10*time.Millisecond)
}

// runGC starts r.run in the background and returns a function to stop it and observe its result.
func runGC(t *testing.T, r *remoteEndpointGC) (stop func()) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- r.run(ctx) }()
	return func() {
		cancel()
		select {
		case err := <-done:
			require.NoError(t, err)
		case <-time.After(2 * time.Second):
			t.Fatal("timed out waiting for run() to stop")
		}
	}
}

func TestRun_CiliumEndpoint(t *testing.T) {
	gc := &recordingGCRunner{}
	r := newTestRemoteEndpointGC(t, gc, false)
	fake := newFakeResource[*types.CiliumEndpoint]()
	r.ciliumEndpoint = fake

	stop := runGC(t, r)
	defer stop()

	// Upsert events must be ignored entirely, even for a remote endpoint.
	send(t, fake, resource.Upsert, resource.Key{Name: "remote"}, &types.CiliumEndpoint{
		Networking: networking(remoteNodeIP, "10.1.0.9", ""),
	})

	// Deletion of an endpoint on the local node must not GC anything.
	send(t, fake, resource.Delete, resource.Key{Name: "local"}, &types.CiliumEndpoint{
		Networking: networking(localNodeIP, "10.1.0.1", ""),
	})

	// Deletion of a remote endpoint must trigger a GC run for its addresses.
	send(t, fake, resource.Delete, resource.Key{Name: "remote"}, &types.CiliumEndpoint{
		Networking: networking(remoteNodeIP, "10.1.0.9", ""),
	})

	require.Eventually(t, func() bool { return len(gc.snapshot()) > 0 }, 2*time.Second, 5*time.Millisecond)
	calls := gc.snapshot()
	require.Len(t, calls, 1)
	require.Equal(t, map[ctmap.NetAddr]struct{}{addrOf(t, "10.1.0.9"): {}}, calls[0].MatchIPs)
}

func TestRun_CiliumEndpointSlice(t *testing.T) {
	gc := &recordingGCRunner{}
	r := newTestRemoteEndpointGC(t, gc, true)
	fake := newFakeResource[*cilium_v2a1.CiliumEndpointSlice]()
	r.ciliumEndpointSlice = fake

	stop := runGC(t, r)
	defer stop()

	key := resource.Key{Name: "ces-1"}
	epA := cilium_v2a1.CoreCiliumEndpoint{Name: "a", Networking: networking(remoteNodeIP, "10.1.0.1", "")}
	epB := cilium_v2a1.CoreCiliumEndpoint{Name: "b", Networking: networking(remoteNodeIP, "10.1.0.2", "")}
	epLocal := cilium_v2a1.CoreCiliumEndpoint{Name: "local", Networking: networking(localNodeIP, "10.1.0.3", "")}

	// The first Upsert observed for a slice is the initial listing: since nothing was
	// previously known about it, nothing must be GC'd, even though endpoints are present.
	send(t, fake, resource.Upsert, key, &cilium_v2a1.CiliumEndpointSlice{
		Endpoints: []cilium_v2a1.CoreCiliumEndpoint{epA, epB, epLocal},
	})
	require.Never(t, func() bool { return len(gc.snapshot()) > 0 }, 200*time.Millisecond, 10*time.Millisecond,
		"initial sync of a CiliumEndpointSlice must not GC live endpoints")

	// "a" is removed from the slice on this update: only its address must be GC'd.
	send(t, fake, resource.Upsert, key, &cilium_v2a1.CiliumEndpointSlice{
		Endpoints: []cilium_v2a1.CoreCiliumEndpoint{epB, epLocal},
	})
	require.Eventually(t, func() bool { return len(gc.snapshot()) > 0 }, 2*time.Second, 5*time.Millisecond)
	calls := gc.snapshot()
	require.Len(t, calls, 1)
	require.Equal(t, map[ctmap.NetAddr]struct{}{addrOf(t, "10.1.0.1"): {}}, calls[0].MatchIPs)

	// The whole slice is deleted: its remaining remote endpoint ("b") must be GC'd,
	// while the endpoint on the local node must still be excluded.
	send(t, fake, resource.Delete, key, &cilium_v2a1.CiliumEndpointSlice{
		Endpoints: []cilium_v2a1.CoreCiliumEndpoint{epB, epLocal},
	})
	require.Eventually(t, func() bool { return len(gc.snapshot()) > 1 }, 2*time.Second, 5*time.Millisecond)
	calls = gc.snapshot()
	require.Len(t, calls, 2)
	require.Equal(t, map[ctmap.NetAddr]struct{}{addrOf(t, "10.1.0.2"): {}}, calls[1].MatchIPs)
}
