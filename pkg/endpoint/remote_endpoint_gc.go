// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package endpoint

import (
	"context"
	"fmt"
	"log/slog"
	"net/netip"

	v2 "github.com/cilium/cilium/pkg/k8s/apis/cilium.io/v2"
	cilium_v2a1 "github.com/cilium/cilium/pkg/k8s/apis/cilium.io/v2alpha1"

	"github.com/cilium/cilium/pkg/k8s/resource"
	"github.com/cilium/cilium/pkg/k8s/types"
	"github.com/cilium/cilium/pkg/lock"
	"github.com/cilium/cilium/pkg/logging/logfields"
	"github.com/cilium/cilium/pkg/maps/ctmap"
	"github.com/cilium/cilium/pkg/node"
	"github.com/cilium/cilium/pkg/option"
	"github.com/cilium/cilium/pkg/time"
	"github.com/cilium/cilium/pkg/trigger"
	"github.com/cilium/hive/cell"
	"github.com/cilium/hive/job"
	"github.com/spf13/pflag"
)

var RemoteEndpointGCCell = cell.Module(
	"remote-endpoint-ct-gc",
	"Garbage collection of CT and NAT maps on remote endpoints",
	cell.Config(defaultConfig),
	cell.Invoke(registerRemoteEndpointGC),
)

type Config struct {
	CiliumRemoteEndpointCTGCBatchingInterval time.Duration
}

var defaultConfig = Config{
	CiliumRemoteEndpointCTGCBatchingInterval: 2 * time.Second,
}

func (def Config) Flags(flags *pflag.FlagSet) {
	flags.Duration(
		"cilium-remote-endpoint-ct-gc-batching-interval",
		def.CiliumRemoteEndpointCTGCBatchingInterval,
		"Time between triggers of CT/NAT GC batch invocations",
	)
}

type remoteEndpointGC struct {
	log                        *slog.Logger
	ctMapGC                    ctmap.GCRunner
	localNode                  *node.LocalNodeStore
	ciliumEndpoint             resource.Resource[*types.CiliumEndpoint]
	ciliumEndpointSlice        resource.Resource[*cilium_v2a1.CiliumEndpointSlice]
	mu                         lock.Mutex
	gcBatch                    map[ctmap.NetAddr]struct{}
	trigger                    *trigger.Trigger
	ciliumEndpointSliceEnabled bool
}

type remoteEndpointGCParams struct {
	cell.In

	Config              Config
	Logger              *slog.Logger
	JobGroup            job.Group
	CiliumEndpoint      resource.Resource[*types.CiliumEndpoint]
	CiliumEndpointSlice resource.Resource[*cilium_v2a1.CiliumEndpointSlice]
	LocalNodeStore      *node.LocalNodeStore
	CTMapGC             ctmap.GCRunner
	DaemonCfg           *option.DaemonConfig
}

func registerRemoteEndpointGC(p remoteEndpointGCParams) error {
	r := &remoteEndpointGC{
		log:                        p.Logger,
		ctMapGC:                    p.CTMapGC,
		localNode:                  p.LocalNodeStore,
		ciliumEndpointSlice:        p.CiliumEndpointSlice,
		ciliumEndpoint:             p.CiliumEndpoint,
		ciliumEndpointSliceEnabled: p.DaemonCfg.EnableCiliumEndpointSlice,
		gcBatch:                    map[ctmap.NetAddr]struct{}{},
	}

	t, err := trigger.NewTrigger(trigger.Parameters{
		Name:        "remote-endpoint-ct-gc-batching",
		MinInterval: max(time.Second, p.Config.CiliumRemoteEndpointCTGCBatchingInterval),
		TriggerFunc: func(reasons []string) {
			r.mu.Lock()
			batch := r.gcBatch
			r.gcBatch = map[ctmap.NetAddr]struct{}{}
			r.mu.Unlock()

			if len(batch) == 0 {
				return
			}

			deleted, err := r.ctMapGC.Run(ctmap.GCFilter{MatchIPs: batch})
			if err != nil {
				r.log.Warn("remote endpoint CT/NAT GC failed", logfields.Error, err)
				return
			}
			r.log.Debug("remote endpoint CT/NAT GC deleted", logfields.Count, deleted, "batch", len(batch))
		},
	})
	if err != nil {
		return err
	}
	r.trigger = t

	p.JobGroup.Add(job.OneShot("remote-endpoint-ct-gc", func(ctx context.Context, health cell.Health) error {
		return r.run(ctx)
	}))

	return nil
}

func (r *remoteEndpointGC) onDelete(addrs map[ctmap.NetAddr]struct{}) {
	if len(addrs) == 0 {
		return
	}

	r.mu.Lock()

	for ip := range addrs {
		r.gcBatch[ip] = struct{}{}
	}

	r.mu.Unlock()
	r.trigger.TriggerWithReason("endpoint(s) deleted")
}

func (r *remoteEndpointGC) run(ctx context.Context) error {
	defer r.trigger.Shutdown()

	ln, err := r.localNode.Get(ctx)

	if err != nil {
		return fmt.Errorf("get local node: %w", err)
	}

	localNodeIP := node.GetCiliumEndpointNodeIP(ln)

	if r.ciliumEndpointSliceEnabled {
		events := r.ciliumEndpointSlice.Events(ctx)
		cache := make(map[resource.Key]*cilium_v2a1.CiliumEndpointSlice)

		for {
			select {
			case <-ctx.Done():
				return nil

			case ev, ok := <-events:
				if !ok {
					return nil
				}

				switch ev.Kind {
				case resource.Upsert:
					removed := removedCESEndpoints(cache[ev.Key], ev.Object)
					cache[ev.Key] = ev.Object
					r.onDelete(r.endpointAddrs(remoteNetworkings(removed, localNodeIP)))
				case resource.Delete:
					removed := ev.Object.Endpoints
					delete(cache, ev.Key)
					r.onDelete(r.endpointAddrs(remoteNetworkings(removed, localNodeIP)))
				}
				ev.Done(nil)
			}
		}
	} else {
		events := r.ciliumEndpoint.Events(ctx)

		for {
			select {
			case <-ctx.Done():

				return nil

			case ev, ok := <-events:
				if !ok {
					return nil
				}
				if ev.Kind == resource.Delete &&
					ev.Object.Networking != nil &&
					ev.Object.Networking.NodeIP != localNodeIP {

					r.onDelete(r.endpointAddrs([]*v2.EndpointNetworking{ev.Object.Networking}))
				}
				ev.Done(nil)
			}
		}
	}
}

func (r *remoteEndpointGC) endpointAddrs(networks []*v2.EndpointNetworking) map[ctmap.NetAddr]struct{} {
	addrs := map[ctmap.NetAddr]struct{}{}

	for _, net := range networks {
		for _, pair := range net.Addressing {
			if pair.IPV4 != "" {
				if a, err := netip.ParseAddr(pair.IPV4); err != nil {
					r.log.Error("parsing IPV4 address", logfields.Error, err)
				} else {
					addrs[ctmap.NetAddr{Addr: a}] = struct{}{}
				}
			}
			if pair.IPV6 != "" {
				if a, err := netip.ParseAddr(pair.IPV6); err != nil {
					r.log.Error("parsing IPV6 address", logfields.Error, err)
				} else {
					addrs[ctmap.NetAddr{Addr: a}] = struct{}{}
				}
			}
		}
	}

	return addrs
}

// removedCESEndpoints returns the endpoints present in old but no longer present in new,
// identified by name. old may be nil, e.g. when new is the first observed version of the slice.
func removedCESEndpoints(old, new *cilium_v2a1.CiliumEndpointSlice) []cilium_v2a1.CoreCiliumEndpoint {
	if old == nil {
		return nil
	}

	present := make(map[string]struct{}, len(new.Endpoints))
	for _, ep := range new.Endpoints {
		present[ep.Name] = struct{}{}
	}

	var removed []cilium_v2a1.CoreCiliumEndpoint
	for _, ep := range old.Endpoints {
		if _, ok := present[ep.Name]; !ok {
			removed = append(removed, ep)
		}
	}

	return removed
}

// remoteNetworkings returns the networking of the given endpoints, excluding those without
// networking information and those running on localNodeIP.
func remoteNetworkings(ceps []cilium_v2a1.CoreCiliumEndpoint, localNodeIP string) []*v2.EndpointNetworking {
	networks := make([]*v2.EndpointNetworking, 0, len(ceps))

	for _, ep := range ceps {
		if ep.Networking != nil && ep.Networking.NodeIP != localNodeIP {
			networks = append(networks, ep.Networking)
		}
	}

	return networks
}
