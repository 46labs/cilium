// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package pinning

import (
	"log/slog"
	"net/netip"
	"testing"

	"github.com/cilium/cilium/pkg/hive"
	"github.com/cilium/cilium/pkg/k8s/resource"
	slim_corev1 "github.com/cilium/cilium/pkg/k8s/slim/k8s/api/core/v1"
	"github.com/cilium/hive/cell"
	"github.com/cilium/hive/hivetest"
	"github.com/cilium/hive/job"
	"github.com/stretchr/testify/require"
)

func setupAgent(t *testing.T) *PinningManagerTestSuite {
	logger := hivetest.Logger(t, hivetest.LogLevel(slog.LevelInfo))

	var jg job.Group
	h := hive.New(
		cell.Invoke(func(g job.Group) { jg = g }),
	)

	require.NoError(t, h.Start(logger, t.Context()))
	t.Cleanup(func() { h.Stop(logger, t.Context()) })

	agent := NewPinningManagerTestSuite(t, node1, node1Ip, jg, logger, NewLbPinMapEventStream())
	go agent.handlePinMapUpdates()
	agent.services.sync(t)
	agent.nodes.sync(t)

	return agent
}

func TestPinningManager(t *testing.T) {
	logger := hivetest.Logger(t, hivetest.LogLevel(slog.LevelInfo))

	var jg job.Group
	h := hive.New(
		cell.Invoke(func(g job.Group) { jg = g }),
	)

	require.NoError(t, h.Start(logger, t.Context()))
	t.Cleanup(func() { h.Stop(logger, t.Context()) })

	expectedNodeCount := len(allNodes)
	expectedServiceCount := len(AllServices)

	agent1 := NewPinningManagerTestSuite(t, node1, node1Ip, jg, logger, NewLbPinMapEventStream())

	nodes1 := []NodeInfo{
		{Name: node2, Ip: node2Ip},
		{Name: node3, Ip: node3Ip},
	}

	agent1.Init(t, nodes1, expectedNodeCount, AllServices, expectedServiceCount, pinToNode)

	agent2 := NewPinningManagerTestSuite(t, node2, node2Ip, jg, logger, NewLbPinMapEventStream())
	nodes2 := []NodeInfo{
		{Name: node1, Ip: node1Ip},
		{Name: node3, Ip: node3Ip},
	}
	agent2.Init(t, nodes2, expectedNodeCount, AllServices, expectedServiceCount, pinToNode)

	agent3 := NewPinningManagerTestSuite(t, node3, node3Ip, jg, logger, NewLbPinMapEventStream())

	nodes3 := []NodeInfo{
		{Name: node1, Ip: node1Ip},
		{Name: node2, Ip: node2Ip},
	}

	agent3.Init(t, nodes3, expectedNodeCount, AllServices, expectedServiceCount, pinToNode)

	require.Equal(t, expectedNodeCount, len(agent1.manager.nodesCache))
	require.Equal(t, expectedNodeCount, len(agent2.manager.nodesCache))
	require.Equal(t, expectedNodeCount, len(agent3.manager.nodesCache))

	require.Equal(t, expectedServiceCount, len(agent1.manager.servicesCache))
	require.Equal(t, expectedServiceCount, len(agent2.manager.servicesCache))
	require.Equal(t, expectedServiceCount, len(agent3.manager.servicesCache))

	agent1.checkPinningMap(t)
	agent2.checkPinningMap(t)
	agent3.checkPinningMap(t)
}

func TestPinningServiceLifecycle(t *testing.T) {
	agent := setupAgent(t)
	agent.addNode(t, node2, node2Ip, readyCond)
	waitForCachedEntities(t, "nodes", agent.manager.nodesCache, 2)

	svc := newK8sPinnedService(service1, Service1Ip, node2, serviceAppSelector)

	// 1. Upsert without annotation — no pinning expected
	delete(svc.Annotations, "service.cilium.io/svc-pinning-node")
	agent.services.process(t, resource.Event[*slim_corev1.Service]{
		Key:    resource.Key{Name: service1},
		Kind:   resource.Upsert,
		Object: svc,
	})
	waitForCachedEntities(t, "services", agent.manager.servicesCache, 0)

	// 2. Upsert with annotation — service pinned
	svc.Annotations["service.cilium.io/svc-pinning-node"] = node2
	agent.services.process(t, resource.Event[*slim_corev1.Service]{
		Key:    resource.Key{Name: service1},
		Kind:   resource.Upsert,
		Object: svc,
	})
	waitForCachedEntities(t, "services", agent.manager.servicesCache, 1)

	// 3. Upsert without annotation — should clean up
	delete(svc.Annotations, "service.cilium.io/svc-pinning-node")
	agent.services.process(t, resource.Event[*slim_corev1.Service]{
		Key:    resource.Key{Name: service1},
		Kind:   resource.Upsert,
		Object: svc,
	})
	waitForCachedEntities(t, "services", agent.manager.servicesCache, 0)

	// 4. Add it back, then Delete event — should clean up
	svc.Annotations["service.cilium.io/svc-pinning-node"] = node2
	agent.services.process(t, resource.Event[*slim_corev1.Service]{
		Key:    resource.Key{Name: service1},
		Kind:   resource.Upsert,
		Object: svc,
	})
	waitForCachedEntities(t, "services", agent.manager.servicesCache, 1)

	agent.services.process(t, resource.Event[*slim_corev1.Service]{
		Key:    resource.Key{Name: service1},
		Kind:   resource.Delete,
		Object: svc,
	})
	waitForCachedEntities(t, "services", agent.manager.servicesCache, 0)

	// 5. Remove ExternalIPs while keeping the annotation — should clean up
	svc.Spec.ExternalIPs = nil
	agent.services.process(t, resource.Event[*slim_corev1.Service]{
		Key:    resource.Key{Name: service1},
		Kind:   resource.Upsert,
		Object: svc,
	})
	waitForCachedEntities(t, "services", agent.manager.servicesCache, 0)

	// 6. Restore IPv4 ExternalIP — service pinned again
	svc.Spec.ExternalIPs = []string{Service1Ip.String()}
	agent.services.process(t, resource.Event[*slim_corev1.Service]{
		Key:    resource.Key{Name: service1},
		Kind:   resource.Upsert,
		Object: svc,
	})
	waitForCachedEntities(t, "services", agent.manager.servicesCache, 1)

	// 7. Transition to IPv6 ExternalIP while keeping the annotation — should clean up
	svc.Spec.ExternalIPs = []string{"2001:db8::1"}
	agent.services.process(t, resource.Event[*slim_corev1.Service]{
		Key:    resource.Key{Name: service1},
		Kind:   resource.Upsert,
		Object: svc,
	})
	waitForCachedEntities(t, "services", agent.manager.servicesCache, 0)

	// 8. IPv6 ExternalIP for a new service — should not be pinned
	svc6 := newK8sPinnedService(service2, netip.MustParseAddr("2001:db8::1"), node2, serviceAppSelector)
	agent.services.process(t, resource.Event[*slim_corev1.Service]{
		Key:    resource.Key{Name: service2},
		Kind:   resource.Upsert,
		Object: svc6,
	})
	waitForCachedEntities(t, "services", agent.manager.servicesCache, 0)
}

func TestPinningNodeIPv4Loss(t *testing.T) {
	agent := setupAgent(t)

	agent.addNode(t, node2, node2Ip, readyCond)
	waitForCachedEntities(t, "nodes", agent.manager.nodesCache, 2)

	ipv6Node := newK8sNode(node2, netip.MustParseAddr("2001:db8::1"), readyCond)
	agent.nodes.process(t, resource.Event[*slim_corev1.Node]{
		Key:    resource.Key{Name: node2},
		Kind:   resource.Upsert,
		Object: ipv6Node,
	})
	waitForCachedEntities(t, "nodes", agent.manager.nodesCache, 1)
}
