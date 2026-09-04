// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package reflectors

import (
	"net/netip"

	"github.com/cilium/hive/cell"
	"github.com/cilium/hive/job"
	"github.com/cilium/statedb"
	"github.com/cilium/statedb/index"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/cilium/cilium/pkg/annotation"
	"github.com/cilium/cilium/pkg/k8s"
	"github.com/cilium/cilium/pkg/k8s/client"
	slim_corev1 "github.com/cilium/cilium/pkg/k8s/slim/k8s/api/core/v1"
	"github.com/cilium/cilium/pkg/k8s/utils"
)

// LbSrcRangeGroupPod is Cilium's internal model of the LB (by source range group) pods running on all nodes.
type LbSrcRangeGroupPod struct {
	UID          string
	Namespace    string
	Name         string
	IP           netip.Addr
	SourceRanges string
}

func (p LbSrcRangeGroupPod) TableHeader() []string {
	return []string{
		"UID",
		"Name",
		"IP",
		"GroupIndex",
	}
}

func (p LbSrcRangeGroupPod) TableRow() []string {
	return []string{
		string(p.UID),
		p.Namespace + "/" + p.Name,
		p.IP.String(),
		p.SourceRanges,
	}
}

const (
	PodTableName = "lb-source-range-group-pods"
)

func newNameIndex() statedb.Index[LbSrcRangeGroupPod, string] {
	return statedb.Index[LbSrcRangeGroupPod, string]{
		Name: "name",
		FromObject: func(obj LbSrcRangeGroupPod) index.KeySet {
			if ns := obj.Namespace; ns != "" {
				return index.NewKeySet(index.String(ns + "/" + obj.Name))
			}
			return index.NewKeySet(index.String(obj.Name))
		},
		FromKey: index.String,
		FromString: func(key string) (index.Key, error) {
			return index.String(key), nil
		},
		Unique: true,
	}
}

func newIpIndex() statedb.Index[LbSrcRangeGroupPod, string] {
	return statedb.Index[LbSrcRangeGroupPod, string]{
		Name: "ip",
		FromObject: func(obj LbSrcRangeGroupPod) index.KeySet {
			return index.NewKeySet(index.String(obj.IP.String()))
		},
		FromKey: index.String,
		FromString: func(key string) (index.Key, error) {
			return index.String(key), nil
		},
		Unique: true,
	}
}

var (
	podNameIndex                = newNameIndex()
	podIpIndex                  = newIpIndex()
	LbSrcRangeGroupPodTableCell = cell.Provide(NewPodTableAndReflector)
)

func NewPodTableAndReflector(jg job.Group, db *statedb.DB, cs client.Clientset) (statedb.Table[LbSrcRangeGroupPod], error) {
	pods, err := NewPodTable(db)
	if err != nil {
		return nil, err
	}

	if !cs.IsEnabled() {
		return pods, nil
	}

	cfg := podReflectorConfig(cs, pods)
	err = k8s.RegisterReflector(jg, db, cfg)
	return pods, err
}

func PodByName(namespace, name string) statedb.Query[LbSrcRangeGroupPod] {
	return podNameIndex.Query(namespace + "/" + name)
}

func PodByIp(ip netip.Addr) statedb.Query[LbSrcRangeGroupPod] {
	return podIpIndex.Query(ip.String())
}

func NewPodTable(db *statedb.DB) (statedb.RWTable[LbSrcRangeGroupPod], error) {
	return statedb.NewTable(
		db,
		PodTableName,
		podNameIndex,
		podIpIndex,
	)
}

func podReflectorConfig(cs client.Clientset, pods statedb.RWTable[LbSrcRangeGroupPod]) k8s.ReflectorConfig[LbSrcRangeGroupPod] {
	lw := utils.ListerWatcherWithModifiers(
		utils.ListerWatcherFromTyped(cs.Slim().CoreV1().Pods("")),
		func(opts *metav1.ListOptions) {
			opts.LabelSelector = annotation.SourceAndPortRangeLbEnabled
		})

	return k8s.ReflectorConfig[LbSrcRangeGroupPod]{
		Name:          "lb-source-range-group",
		Table:         pods,
		ListerWatcher: lw,
		MetricScope:   "Pod",
		Transform: func(_ statedb.ReadTxn, obj any) (LbSrcRangeGroupPod, bool) {
			pod, ok := obj.(*slim_corev1.Pod)
			if !ok {
				return LbSrcRangeGroupPod{}, false
			}

			sourceRanges := pod.Annotations[annotation.PodSourceRanges]

			ip, err := netip.ParseAddr(pod.Status.PodIP)

			if err != nil {
				return LbSrcRangeGroupPod{}, false
			}

			return LbSrcRangeGroupPod{
				UID:          string(pod.UID),
				Namespace:    pod.Namespace,
				Name:         pod.Name,
				IP:           ip,
				SourceRanges: sourceRanges,
			}, true
		},
	}
}
