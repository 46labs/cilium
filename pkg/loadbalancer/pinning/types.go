// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package pinning

import (
	"github.com/cilium/cilium/pkg/loadbalancer/maps"
)

type reconcileUpdate interface {
	isReconcileUpdate()
}

type addService struct {
	ServiceIp string
	ServiceId string
}
type deleteService struct {
	ServiceId string
	ServiceIp string
}
type syncService struct{}
type addNode struct {
	NodeId string
	NodeIp string
}
type deleteNode struct {
	NodeId string
	NodeIp string
}
type syncNode struct{}

func (addService) isReconcileUpdate()    {}
func (deleteService) isReconcileUpdate() {}
func (syncService) isReconcileUpdate()   {}
func (addNode) isReconcileUpdate()       {}
func (deleteNode) isReconcileUpdate()    {}
func (syncNode) isReconcileUpdate()      {}

type nodeIp string
type serviceIp string

type servicesMap map[string]serviceIp
type nodesMap map[string]nodeIp

type pinningMap map[*maps.LbPinning4Key]*maps.LbPinning4Value

// key - ServiceIP, value - NodeIP
type simplePinningMap map[serviceIp]nodeIp

type servicePinner interface {
	makePinningMap(
		originNodeIp string,
		previousServices map[string]serviceIp,
		newServices map[string]serviceIp,
		previousNodes map[string]nodeIp,
		newNodes map[string]nodeIp,
	) (simplePinningMap, error)
}

type LbPinMapUpdateEvent struct{}
