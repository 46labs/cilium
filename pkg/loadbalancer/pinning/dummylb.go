// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package pinning

import (
	"hash/fnv"
	"maps"
	"net/netip"
	"slices"
)

type dummyLbPinning struct{}

func parseIps[T ~string](ips []T) ([]netip.Addr, error) {
	sortedIps := []netip.Addr{}

	for _, ip := range ips {
		ipAddr, err := netip.ParseAddr(string(ip))

		if err != nil {
			return sortedIps, err
		}

		sortedIps = append(sortedIps, ipAddr)
	}

	return sortedIps, nil
}

func parseAndSortIps[T ~string](ips []T) ([]netip.Addr, error) {
	sortedIps, err := parseIps(ips)

	if err != nil {
		return sortedIps, nil
	}

	slices.SortFunc(sortedIps, func(a, b netip.Addr) int {
		return a.Compare(b)
	})

	return sortedIps, nil
}

func calcNodeIndex(ipAddr netip.Addr, nodeCount uint32) int {
	h := fnv.New32a()
	h.Write(ipAddr.AsSlice())

	return int(h.Sum32() % nodeCount)
}

func rebalanceServices(
	m simplePinningMap,
	services map[string]serviceIp,
	nodes []nodeIp,
	excludeNodeIp nodeIp,
) error {
	nodesCount := len(nodes)

	serviceIps, err := parseIps(slices.Collect(maps.Values(services)))

	if err != nil {
		return err
	}

	nodeIps, err := parseAndSortIps(nodes)

	if err != nil {
		return err
	}

	for _, ip := range serviceIps {
		i := calcNodeIndex(ip, uint32(nodesCount))

		nodeip := nodeIp(nodeIps[i].String())

		if nodeip != excludeNodeIp {
			m[serviceIp(ip.String())] = nodeip
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
