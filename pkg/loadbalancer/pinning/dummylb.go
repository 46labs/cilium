package pinning

import (
	"encoding/binary"
	"maps"
	"net/netip"
	"slices"
)

type dummyLbPinning struct{}

func calcNodeIndex(serviceIp string, nodeCount uint32) (int, error) {
	ipAddr, err := netip.ParseAddr(serviceIp)

	if err != nil {
		return -1, err
	}

	fourBytes := ipAddr.As4()

	n := binary.BigEndian.Uint32(fourBytes[:])
	i := n % nodeCount

	return int(i), nil
}

func rebalanceServices(
	m simplePinningMap,
	services map[string]serviceIp,
	nodes []nodeIp,
	excludeNodeIp nodeIp,
) error {
	nodesCount := len(nodes)

	for _, ip := range services {
		i, err := calcNodeIndex(string(ip), uint32(nodesCount))

		if err != nil {
			return err
		}

		println("nodesCount", nodesCount, "i", i, "nodes[i]", string(nodes[i]), "excludeNodeIp", string(excludeNodeIp))
		if nodes[i] != excludeNodeIp {
			m[ip] = nodes[i]
		}
	}

	return nil
}

func (that dummyLbPinning) makePinningMap(
	originNodeIp string,
	previousServices map[string]serviceIp,
	newServices map[string]serviceIp,
	previousNodes map[string]nodeIp,
	newNodes map[string]nodeIp,
) (simplePinningMap, error) {
	m := simplePinningMap{}
	nodes := slices.Collect(maps.Values(newNodes))

	if err := rebalanceServices(
		m,
		newServices,
		nodes,
		nodeIp(originNodeIp),
	); err != nil {
		return nil, err
	}

	return m, nil
}
