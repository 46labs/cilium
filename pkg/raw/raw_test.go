// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package raw

import (
	"context"
	"log/slog"
	"net/netip"
	"testing"

	slim_corev1 "github.com/cilium/cilium/pkg/k8s/slim/k8s/api/core/v1"
	slim_metav1 "github.com/cilium/cilium/pkg/k8s/slim/k8s/apis/meta/v1"
	"github.com/cilium/cilium/pkg/maps/crap"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
)

var testSelector = map[string]string{
	"app": "test",
}

func countValidRules(val crap.CrapVal) int {
	count := 0
	for i := 0; i < crap.MaxRulesPerIP; i++ {
		if val.Rules[i].IsValid() {
			count++
		}
	}
	return count
}

func requireRule(t *testing.T, val crap.CrapVal, portBegin, portEnd uint16, podIP netip.Addr) {
	t.Helper()
	for i := 0; i < crap.MaxRulesPerIP; i++ {
		if !val.Rules[i].IsValid() {
			break
		}
		if val.Rules[i].PortBegin == portBegin && val.Rules[i].PortEnd == portEnd {
			require.Equal(t, podIP, val.Rules[i].PodIp.Addr())
			return
		}
	}
	t.Fatalf("rule for ports %d-%d pointing to %s not found", portBegin, portEnd, podIP)
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
			labels:     testSelector,
			id:         svc1,
			vip:        []netip.Addr{netip.MustParseAddr("20.20.20.1")},
			namespace:  "ns1",
			portRanges: []portRange{{begin: 0, end: 65535}},
		},
		svc2: {
			labels:     testSelector,
			id:         svc2,
			vip:        []netip.Addr{netip.MustParseAddr("20.20.20.2")},
			namespace:  "ns2",
			portRanges: []portRange{{begin: 0, end: 65535}},
		},
	}

	rules := buildRules(eps, svcs)

	require.Equal(t, eps[pod1].ip, rules[crap.CrapKey{DestIP: svcs[svc1].vip[0].As4()}].Rules[0].PodIp.Addr())
	require.Equal(t, eps[pod2].ip, rules[crap.CrapKey{DestIP: svcs[svc2].vip[0].As4()}].Rules[0].PodIp.Addr())
}

func TestAddServiceWithoutAnnotationEmitsDelete(t *testing.T) {
	cm := &CrapManager{
		logger: slog.New(slog.DiscardHandler),
	}

	svc := &slim_corev1.Service{
		ObjectMeta: slim_metav1.ObjectMeta{
			UID:         "service-uid",
			Name:        "raw-svc",
			Namespace:   "default",
			Annotations: map[string]string{},
		},
		Spec: slim_corev1.ServiceSpec{
			ExternalIPs: []string{"192.0.2.1"},
		},
	}

	diff := newDiff()
	cm.addService(context.Background(), svc, diff)

	got, ok := diff.serviceDiff[serviceID(svc.UID)]
	if !ok {
		t.Fatalf("expected a diff event for service %s, got none", svc.UID)
	}
	if got != nil {
		t.Fatalf("expected a delete event (nil value) for service without RAW annotation, got add event: %+v", got)
	}
}

func TestBuildRulesNamespaceIsolation(t *testing.T) {
	svc := &serviceMetadata{
		id:         "svc-uid",
		namespace:  "ns-a",
		labels:     map[string]string{"app": "foo"},
		vip:        []netip.Addr{netip.MustParseAddr("192.0.2.1")},
		portRanges: []portRange{{begin: 0, end: 65535}},
	}

	epSameNs := &endpointMetadata{
		id:        "ep-same-ns",
		namespace: "ns-a",
		ip:        netip.MustParseAddr("10.0.0.1"),
		labels:    map[string]string{"app": "foo"},
	}

	epOtherNs := &endpointMetadata{
		id:        "ep-other-ns",
		namespace: "ns-b",
		ip:        netip.MustParseAddr("10.0.0.2"),
		labels:    map[string]string{"app": "foo"},
	}

	svcs := map[serviceID]*serviceMetadata{svc.id: svc}

	epsSameNs := map[endpointID]*endpointMetadata{epSameNs.id: epSameNs}
	rules := buildRules(epsSameNs, svcs)
	if len(rules) != 1 {
		t.Fatalf("expected 1 rule when endpoint is in same namespace, got %d", len(rules))
	}

	epsOtherNs := map[endpointID]*endpointMetadata{epOtherNs.id: epOtherNs}
	rules = buildRules(epsOtherNs, svcs)
	if len(rules) != 0 {
		t.Fatalf("expected 0 rules when endpoint is in different namespace, got %d", len(rules))
	}

	epsBoth := map[endpointID]*endpointMetadata{
		epSameNs.id:  epSameNs,
		epOtherNs.id: epOtherNs,
	}
	rules = buildRules(epsBoth, svcs)
	if len(rules) != 1 {
		t.Fatalf("expected 1 rule when one endpoint is in same namespace and one in different, got %d", len(rules))
	}
}

func TestParsePortRangeAnnotation(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    []portRange
		wantErr bool
	}{
		{
			name:  "empty string returns default range",
			input: "",
			want:  []portRange{{begin: 0, end: 65535}},
		},
		{
			name:  "single port",
			input: "80",
			want:  []portRange{{begin: 80, end: 80}},
		},
		{
			name:  "port range",
			input: "8000-9000",
			want:  []portRange{{begin: 8000, end: 9000}},
		},
		{
			name:  "multiple comma-separated ranges",
			input: "80,443,3000-4000",
			want:  []portRange{{begin: 80, end: 80}, {begin: 443, end: 443}, {begin: 3000, end: 4000}},
		},
		{
			name:    "begin > end",
			input:   "9000-8000",
			wantErr: true,
		},
		{
			name:    "invalid port string",
			input:   "abc",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parsePortRangeAnnotation(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("parsePortRangeAnnotation() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				require.Equal(t, tt.want, got)
			}
		})
	}
}

func TestBuildRulesMultipleRangesSameService(t *testing.T) {
	ep := &endpointMetadata{
		id:        "ep",
		namespace: "ns",
		ip:        netip.MustParseAddr("10.0.0.1"),
		labels:    map[string]string{"app": "foo"},
	}

	svc := &serviceMetadata{
		id:        "svc",
		namespace: "ns",
		labels:    map[string]string{"app": "foo"},
		vip:       []netip.Addr{netip.MustParseAddr("20.20.20.1")},
		portRanges: []portRange{
			{begin: 80, end: 80},
			{begin: 443, end: 443},
			{begin: 3000, end: 4000},
		},
	}

	rules := buildRules(
		map[endpointID]*endpointMetadata{ep.id: ep},
		map[serviceID]*serviceMetadata{svc.id: svc},
	)

	key := crap.NewKey(netip.MustParseAddr("20.20.20.1"))
	val, ok := rules[key]
	if !ok {
		t.Fatal("expected a rule for 20.20.20.1")
	}

	seenPorts := make(map[[2]uint16]bool)
	for i := 0; i < crap.MaxRulesPerIP; i++ {
		if !val.Rules[i].IsValid() {
			break
		}
		if val.Rules[i].PodIp.Addr() != ep.ip {
			t.Errorf("rule[%d] pod IP = %s, want %s", i, val.Rules[i].PodIp.Addr(), ep.ip)
		}
		seenPorts[[2]uint16{val.Rules[i].PortBegin, val.Rules[i].PortEnd}] = true
	}

	for _, pr := range svc.portRanges {
		if !seenPorts[[2]uint16{pr.begin, pr.end}] {
			t.Errorf("missing port range %d-%d in rules", pr.begin, pr.end)
		}
	}

	if len(seenPorts) != len(svc.portRanges) {
		t.Errorf("expected %d rules, got %d", len(svc.portRanges), len(seenPorts))
	}
}

func TestBuildRulesMultipleServicesSameVIP(t *testing.T) {
	ep1 := &endpointMetadata{
		id:        "ep1",
		namespace: "ns",
		ip:        netip.MustParseAddr("10.0.0.2"),
		labels:    map[string]string{"app": "one"},
	}
	ep2 := &endpointMetadata{
		id:        "ep2",
		namespace: "ns",
		ip:        netip.MustParseAddr("10.0.0.1"),
		labels:    map[string]string{"app": "two"},
	}
	ep3 := &endpointMetadata{
		id:        "ep3",
		namespace: "ns",
		ip:        netip.MustParseAddr("2001:db8::1"),
		labels:    map[string]string{"app": "three"},
	}

	sharedVIP := netip.MustParseAddr("20.20.20.1")

	svc1 := &serviceMetadata{
		id:         "svc1",
		namespace:  "ns",
		labels:     map[string]string{"app": "one"},
		vip:        []netip.Addr{sharedVIP},
		portRanges: []portRange{{begin: 80, end: 80}},
	}
	svc2 := &serviceMetadata{
		id:         "svc2",
		namespace:  "ns",
		labels:     map[string]string{"app": "two"},
		vip:        []netip.Addr{sharedVIP},
		portRanges: []portRange{{begin: 443, end: 443}},
	}
	svc3 := &serviceMetadata{
		id:         "svc3",
		namespace:  "ns",
		labels:     map[string]string{"app": "three"},
		vip:        []netip.Addr{sharedVIP},
		portRanges: []portRange{{begin: 8080, end: 8080}},
	}

	rules := buildRules(
		map[endpointID]*endpointMetadata{ep1.id: ep1, ep2.id: ep2, ep3.id: ep3},
		map[serviceID]*serviceMetadata{svc1.id: svc1, svc2.id: svc2, svc3.id: svc3},
	)

	key := crap.NewKey(sharedVIP)
	val, ok := rules[key]
	require.True(t, ok, "expected a rule for shared VIP")

	require.Equal(t, 2, countValidRules(val))
	requireRule(t, val, 80, 80, ep1.ip)
	requireRule(t, val, 443, 443, ep2.ip)

	require.Equal(t, ep2.ip, val.Rules[0].PodIp.Addr())
	require.Equal(t, ep1.ip, val.Rules[1].PodIp.Addr())
}

func TestBuildRulesMaxRulesLimit(t *testing.T) {
	ep := &endpointMetadata{
		id:        "ep",
		namespace: "ns",
		ip:        netip.MustParseAddr("10.0.0.1"),
		labels:    map[string]string{"app": "foo"},
	}

	ranges := make([]portRange, crap.MaxRulesPerIP+5)
	for i := range ranges {
		ranges[i] = portRange{begin: uint16(1000 + i), end: uint16(1000 + i)}
	}

	svc := &serviceMetadata{
		id:         "svc",
		namespace:  "ns",
		labels:     map[string]string{"app": "foo"},
		vip:        []netip.Addr{netip.MustParseAddr("20.20.20.1")},
		portRanges: ranges,
	}

	rules := buildRules(
		map[endpointID]*endpointMetadata{ep.id: ep},
		map[serviceID]*serviceMetadata{svc.id: svc},
	)

	key := crap.NewKey(netip.MustParseAddr("20.20.20.1"))
	val, ok := rules[key]
	if !ok {
		t.Fatal("expected a rule for 20.20.20.1")
	}

	if count := countValidRules(val); count != crap.MaxRulesPerIP {
		t.Errorf("expected exactly %d rules, got %d", crap.MaxRulesPerIP, count)
	}
}

func TestBuildRulesServiceRemovedNoEntry(t *testing.T) {
	rules := buildRules(
		map[endpointID]*endpointMetadata{},
		map[serviceID]*serviceMetadata{},
	)
	if len(rules) != 0 {
		t.Fatal("expected no rules when no services are present")
	}
}

func TestBuildRulesPartialRuleRemoval(t *testing.T) {
	ep := &endpointMetadata{
		id:        "ep",
		namespace: "ns",
		ip:        netip.MustParseAddr("10.0.0.1"),
		labels:    map[string]string{"app": "foo"},
	}

	svc := &serviceMetadata{
		id:        "svc",
		namespace: "ns",
		labels:    map[string]string{"app": "foo"},
		vip:       []netip.Addr{netip.MustParseAddr("20.20.20.1")},
		portRanges: []portRange{
			{begin: 80, end: 80},
			{begin: 443, end: 443},
		},
	}

	rules := buildRules(
		map[endpointID]*endpointMetadata{ep.id: ep},
		map[serviceID]*serviceMetadata{svc.id: svc},
	)

	key := crap.NewKey(netip.MustParseAddr("20.20.20.1"))
	val, ok := rules[key]
	if !ok {
		t.Fatal("expected entry for VIP")
	}

	if count := countValidRules(val); count != 2 {
		t.Fatalf("expected 2 rules, got %d", count)
	}

	svc.portRanges = []portRange{{begin: 80, end: 80}}

	rules = buildRules(
		map[endpointID]*endpointMetadata{ep.id: ep},
		map[serviceID]*serviceMetadata{svc.id: svc},
	)

	val, ok = rules[key]
	if !ok {
		t.Fatal("expected entry for VIP to still exist")
	}

	if count := countValidRules(val); count != 1 {
		t.Fatalf("expected 1 rule after removal, got %d", count)
	}
	if val.Rules[0].PortBegin != 80 || val.Rules[0].PortEnd != 80 {
		t.Errorf("remaining rule should be port 80, got %d-%d", val.Rules[0].PortBegin, val.Rules[0].PortEnd)
	}
}

func TestBuildRulesEndpointRemovedPartialRules(t *testing.T) {
	ep1 := &endpointMetadata{
		id:        "ep1",
		namespace: "ns",
		ip:        netip.MustParseAddr("10.0.0.10"),
		labels:    map[string]string{"app": "one"},
	}
	ep2 := &endpointMetadata{
		id:        "ep2",
		namespace: "ns",
		ip:        netip.MustParseAddr("10.0.0.20"),
		labels:    map[string]string{"app": "two"},
	}

	sharedVIP := netip.MustParseAddr("20.20.20.1")

	svc1 := &serviceMetadata{
		id:         "svc1",
		namespace:  "ns",
		labels:     map[string]string{"app": "one"},
		vip:        []netip.Addr{sharedVIP},
		portRanges: []portRange{{begin: 80, end: 80}},
	}
	svc2 := &serviceMetadata{
		id:         "svc2",
		namespace:  "ns",
		labels:     map[string]string{"app": "two"},
		vip:        []netip.Addr{sharedVIP},
		portRanges: []portRange{{begin: 443, end: 443}},
	}

	rules := buildRules(
		map[endpointID]*endpointMetadata{ep1.id: ep1, ep2.id: ep2},
		map[serviceID]*serviceMetadata{svc1.id: svc1, svc2.id: svc2},
	)
	key := crap.NewKey(sharedVIP)
	val, ok := rules[key]
	if !ok {
		t.Fatal("expected entry for shared VIP")
	}
	if count := countValidRules(val); count != 2 {
		t.Fatalf("expected 2 rules with both endpoints, got %d", count)
	}

	rules = buildRules(
		map[endpointID]*endpointMetadata{ep1.id: ep1},
		map[serviceID]*serviceMetadata{svc1.id: svc1, svc2.id: svc2},
	)
	val, ok = rules[key]
	if !ok {
		t.Fatal("expected entry for shared VIP to still exist")
	}
	if count := countValidRules(val); count != 1 {
		t.Fatalf("expected 1 rule after ep2 removal, got %d", count)
	}
	if val.Rules[0].PodIp.Addr() != ep1.ip {
		t.Errorf("remaining rule should point to %s, got %s", ep1.ip, val.Rules[0].PodIp.Addr())
	}
}
