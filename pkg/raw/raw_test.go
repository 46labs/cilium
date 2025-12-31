// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package raw

import (
	"context"
	"log/slog"
	"net/netip"
	"testing"

	"github.com/cilium/cilium/pkg/maps/crap"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	slim_corev1 "github.com/cilium/cilium/pkg/k8s/slim/k8s/api/core/v1"
	slim_metav1 "github.com/cilium/cilium/pkg/k8s/slim/k8s/apis/meta/v1"
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

func TestAddServiceWithoutAnnotationEmitsDelete(t *testing.T) {
	cm := &CrapManager{
		logger: slog.New(slog.DiscardHandler),
	}

	svc := &slim_corev1.Service{
		ObjectMeta: slim_metav1.ObjectMeta{
			UID:             "service-uid",
			Name:            "raw-svc",
			Namespace:       "default",
			Annotations:     map[string]string{},
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
		id:        "svc-uid",
		namespace: "ns-a",
		labels:    map[string]string{"app": "foo"},
		vip:       []netip.Addr{netip.MustParseAddr("192.0.2.1")},
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

	// Endpoint in the same namespace should match.
	epsSameNs := map[endpointID]*endpointMetadata{epSameNs.id: epSameNs}
	rules := buildRules(epsSameNs, svcs)
	if len(rules) != 1 {
		t.Fatalf("expected 1 rule when endpoint is in same namespace, got %d", len(rules))
	}

	// Endpoint in a different namespace with the same labels should NOT match.
	epsOtherNs := map[endpointID]*endpointMetadata{epOtherNs.id: epOtherNs}
	rules = buildRules(epsOtherNs, svcs)
	if len(rules) != 0 {
		t.Fatalf("expected 0 rules when endpoint is in different namespace, got %d", len(rules))
	}

	// Both endpoints present: only same-namespace one should match.
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
		name      string
		input     string
		wantBegin uint16
		wantEnd   uint16
		wantErr   bool
	}{
		{
			name:      "empty string returns default range",
			input:     "",
			wantBegin: 0,
			wantEnd:   65535,
		},
		{
			name:      "single port",
			input:     "80",
			wantBegin: 80,
			wantEnd:   80,
		},
		{
			name:      "port range",
			input:     "8000-9000",
			wantBegin: 8000,
			wantEnd:   9000,
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
			begin, end, err := parsePortRangeAnnotation(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("parsePortRangeAnnotation() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && (begin != tt.wantBegin || end != tt.wantEnd) {
				t.Errorf("parsePortRangeAnnotation() = (%d, %d), want (%d, %d)", begin, end, tt.wantBegin, tt.wantEnd)
			}
		})
	}
}
