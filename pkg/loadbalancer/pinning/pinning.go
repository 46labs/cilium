// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package pinning

import (
	"context"
	"log/slog"
	"maps"
	"net/netip"

	"github.com/cilium/cilium/pkg/annotation"
	"github.com/cilium/cilium/pkg/k8s/resource"
	slim_corev1 "github.com/cilium/cilium/pkg/k8s/slim/k8s/api/core/v1"
	lbmaps "github.com/cilium/cilium/pkg/loadbalancer/maps"
	"github.com/cilium/cilium/pkg/logging/logfields"
	"github.com/cilium/cilium/pkg/node"
	"github.com/cilium/hive/cell"
	"github.com/cilium/hive/job"
)

type PinningParams struct {
	cell.In
	JobGroup           job.Group
	Logger             *slog.Logger
	Services           resource.Resource[*slim_corev1.Service]
	Nodes              resource.Resource[*slim_corev1.Node]
	LocalNodeStore     *node.LocalNodeStore
	LBMaps             lbmaps.LBMaps
	PinMapUpdateStream *lbPinMapEventStream
}

type PinningManager struct {
	logger                  *slog.Logger
	services                resource.Resource[*slim_corev1.Service]
	nodes                   resource.Resource[*slim_corev1.Node]
	localNodeStore          *node.LocalNodeStore
	lBMaps                  lbmaps.LBMaps
	reconcileChannel        chan reconcileUpdate
	nodesCache              nodesMap
	servicesCache           servicesMap
	servicePinner           servicePinner
	pinMapUpdateEventStream *lbPinMapEventStream
}

func newPinningManager(params PinningParams) *PinningManager {
	return &PinningManager{
		logger:                  params.Logger,
		services:                params.Services,
		nodes:                   params.Nodes,
		lBMaps:                  params.LBMaps,
		localNodeStore:          params.LocalNodeStore,
		reconcileChannel:        make(chan reconcileUpdate),
		nodesCache:              nodesMap{},
		servicesCache:           servicesMap{},
		servicePinner:           dummyLbPinning{},
		pinMapUpdateEventStream: params.PinMapUpdateStream,
	}
}

func isPinnedService(svc *slim_corev1.Service) bool {
	_, pinned := svc.Annotations[annotation.ServiceSipPinning]

	return pinned
}

func (pm *PinningManager) handleServiceEvent(ctx context.Context, event resource.Event[*slim_corev1.Service]) error {
	var msg reconcileUpdate

	switch event.Kind {
	case resource.Sync:
		msg = syncService{}
	case resource.Delete, resource.Upsert:
		if !isPinnedService(event.Object) {
			return eventDone(event, nil)
		}

		if len(event.Object.Spec.ExternalIPs) > 0 {
			serviceIp := event.Object.Spec.ExternalIPs[0]
			serviceId := string(event.Object.UID)

			if event.Kind == resource.Upsert {
				msg = addService{
					ServiceId: serviceId,
					ServiceIp: serviceIp,
				}
			} else {
				msg = deleteService{
					ServiceId: serviceId,
					ServiceIp: serviceIp,
				}
			}
		}
	}

	event.Done(nil)
	pm.reconcileChannel <- msg

	return nil
}

func isNodeReady(node *slim_corev1.Node) bool {
	for _, cond := range node.Status.Conditions {
		if cond.Type == slim_corev1.NodeReady && cond.Status == slim_corev1.ConditionTrue {
			return true
		}
	}

	return false
}

func (pm *PinningManager) handleNodeEvent(ctx context.Context, event resource.Event[*slim_corev1.Node]) error {
	var msg reconcileUpdate

	switch event.Kind {
	case resource.Sync:
		msg = syncNode{}
	case resource.Upsert, resource.Delete:
		nodeIp := ""

		for _, a := range event.Object.Status.Addresses {
			if a.Type == slim_corev1.NodeInternalIP {
				addr, err := netip.ParseAddr(a.Address)

				if err != nil {
					return eventDone(event, err)
				}

				if addr.Is4() {
					nodeIp = addr.String()
					break
				}
			}
		}

		if nodeIp == "" {
			pm.logger.Error("IP has not found for", logfields.NodeName, event.Key.Name)
			return eventDone(event, nil)
		}

		nodeId := string(event.Object.UID)

		if event.Kind == resource.Upsert && isNodeReady(event.Object) {
			msg = addNode{
				NodeId: nodeId,
				NodeIp: nodeIp,
			}
		} else {
			msg = deleteNode{
				NodeId: nodeId,
				NodeIp: nodeIp,
			}
		}
	}

	event.Done(nil)
	pm.reconcileChannel <- msg

	return nil
}

func (pm *PinningManager) applyPinningMap(desiredPinningMap simplePinningMap) error {
	current, err := DumpPinningMap(pm.lBMaps)

	if err != nil {
		return err
	}

	for k := range *current {
		if err := pm.lBMaps.DeletePinning4(k); err != nil {
			return err
		}
	}

	desired, err := desiredPinningMap.toPinningMap()

	if err != nil {
		return err
	}

	for k, v := range desired {
		if err := pm.lBMaps.UpdatePinning4(k, v); err != nil {
			return err
		}
	}

	return nil
}

func (pm *PinningManager) reconcileLoop(ctx context.Context, health cell.Health) error {
	localNode, err := pm.localNodeStore.Get(ctx)

	if err != nil {
		return err
	}

	defer pm.pinMapUpdateEventStream.complete(err)

	localNodeIp := localNode.GetNodeIP(false).String()

	for {
		select {
		case event := <-pm.reconcileChannel:
			servicesCache := maps.Clone(pm.servicesCache)
			nodesCache := maps.Clone(pm.nodesCache)

			switch msg := event.(type) {
			case addNode:
				pm.nodesCache[msg.NodeId] = nodeIp(msg.NodeIp)
			case deleteNode:
				delete(pm.nodesCache, msg.NodeId)

			case addService:
				pm.servicesCache[msg.ServiceId] = serviceIp(msg.ServiceIp)
			case deleteService:
				delete(pm.servicesCache, msg.ServiceId)

			case syncService, syncNode:
				continue
			}

			desiredPinningMap, err := pm.servicePinner.makePinningMap(
				localNodeIp,
				servicesCache,
				pm.servicesCache,
				nodesCache,
				pm.nodesCache,
			)

			if err != nil {
				pm.logger.Error("error making desired pinning map", logfields.Error, err)
				continue
			}

			if err := pm.applyPinningMap(desiredPinningMap); err != nil {
				pm.logger.Error("error applying desired pinning map", logfields.Error, err)
				continue
			}

			pm.pinMapUpdateEventStream.emitter(LbPinMapUpdateEvent{})

		case <-ctx.Done():
			// graceful shutdown
			pm.pinMapUpdateEventStream.complete(nil)
			return nil
		}
	}
}

func registerPinningManager(params PinningParams) error {
	mng := newPinningManager(params)

	params.JobGroup.Add(job.Observer(
		"pinning-service-observer",
		mng.handleServiceEvent,
		params.Services,
	))

	params.JobGroup.Add(job.Observer(
		"pinning-node-observer",
		mng.handleNodeEvent,
		params.Nodes,
	))

	params.JobGroup.Add(job.OneShot(
		"pinning-reconciler-loop",
		mng.reconcileLoop,
	))

	return nil
}
