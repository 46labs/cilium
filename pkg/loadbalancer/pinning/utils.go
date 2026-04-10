package pinning

import (
	"net/netip"

	"github.com/cilium/cilium/pkg/k8s/resource"
	lbmaps "github.com/cilium/cilium/pkg/loadbalancer/maps"
	"github.com/cilium/cilium/pkg/types"
	k8sRuntime "k8s.io/apimachinery/pkg/runtime"
)

func eventDone[T k8sRuntime.Object](event resource.Event[T], err error) error {
	event.Done(err)

	return err
}

func DumpPinningMap(lBMaps lbmaps.LBMaps) (*pinningMap, error) {
	m := pinningMap{}

	if err := lBMaps.DumpPinning4(func(lpk *lbmaps.LbPinning4Key, lpv *lbmaps.LbPinning4Value) {
		m[lpk] = lpv
	}); err != nil {
		return nil, err
	}

	return &m, nil
}

func (that simplePinningMap) toPinningMap() (pinningMap, error) {
	m := pinningMap{}

	for serviceIp, nodeIp := range that {
		sIpAddr, err := netip.ParseAddr(string(serviceIp))

		if err != nil {
			return nil, err
		}

		sIp := &types.IPv4{}
		sIp.FromAddr(sIpAddr)

		nIpAddr, err := netip.ParseAddr(string(nodeIp))

		if err != nil {
			return nil, err
		}

		nIp := &types.IPv4{}
		nIp.FromAddr(nIpAddr)

		m[&lbmaps.LbPinning4Key{ServiceIP: *sIp}] = &lbmaps.LbPinning4Value{NodeIP: *nIp}
	}

	return m, nil
}

func FromPinningMap(pmap pinningMap) simplePinningMap {
	m := simplePinningMap{}

	for k, v := range pmap {
		m[serviceIp(k.ServiceIP.String())] = nodeIp(v.NodeIP.String())
	}

	return m
}
