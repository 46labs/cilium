package raw

import (
	"net/netip"
	"testing"

	"github.com/cilium/cilium/pkg/maps/crap"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
)

var testSelector = map[string]string{
	"app": "test",
}

func TestRulesPresence(t *testing.T) {
	pod1 := k8stypes.UID("pod_ns1")
	pod2 := k8stypes.UID("pod_ns2")

	svc1 := k8stypes.UID("svc_ns1")
	svc2 := k8stypes.UID("svc_ns2")

	eps := map[endpointID]*endpointMetadata{
		pod1: {
			labels:    testSelector,
			id:        pod1,
			ip:        netip.MustParseAddr("10.10.10.1"),
			nodeIP:    "192.168.1.1",
			namespace: "ns1",
		},
		pod2: {
			labels:    testSelector,
			id:        pod2,
			ip:        netip.MustParseAddr("10.10.10.2"),
			nodeIP:    "192.168.1.1",
			namespace: "ns2",
		},
	}

	svcs := map[serviceID]*serviceMetadata{
		svc1: {
			labels:    testSelector,
			id:        svc1,
			vip:       []netip.Addr{netip.MustParseAddr("20.20.20.1")},
			namespace: "ns1",
		},
		svc2: {
			labels:    testSelector,
			id:        svc2,
			vip:       []netip.Addr{netip.MustParseAddr("20.20.20.2")},
			namespace: "ns2",
		},
	}

	rules := buildRules(eps, svcs)

	require.Equal(t, eps[pod1].ip, rules[crap.CrapKey{DestIP: svcs[svc1].vip[0].As4()}].PodIp.Addr())
	require.Equal(t, eps[pod2].ip, rules[crap.CrapKey{DestIP: svcs[svc2].vip[0].As4()}].PodIp.Addr())
}
