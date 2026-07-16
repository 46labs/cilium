// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package crap

import (
	"net/netip"
	"testing"
)

func TestCrapRuleIsValid(t *testing.T) {
	tests := []struct {
		name  string
		rule  CrapRule
		valid bool
	}{
		{
			name:  "zero value is invalid",
			rule:  CrapRule{},
			valid: false,
		},
		{
			name:  "pod_ip set makes it valid",
			rule:  CrapRule{PodIp: [4]byte{10, 0, 0, 1}},
			valid: true,
		},
		{
			name:  "only port set still invalid",
			rule:  CrapRule{PortBegin: 80, PortEnd: 80},
			valid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.rule.IsValid(); got != tt.valid {
				t.Errorf("CrapRule.IsValid() = %v, want %v", got, tt.valid)
			}
		})
	}
}

func TestNewValSingleRule(t *testing.T) {
	podIP := netip.MustParseAddr("10.0.0.1")
	val := NewVal(podIP, 80, 80)

	if !val.Rules[0].IsValid() {
		t.Fatal("expected rule 0 to be valid")
	}
	if got := val.Rules[0].PodIp.Addr(); got != podIP {
		t.Errorf("rule[0] PodIp = %s, want %s", got, podIP)
	}
	if val.Rules[0].PortBegin != 80 || val.Rules[0].PortEnd != 80 {
		t.Errorf("rule[0] ports = %d-%d, want 80-80", val.Rules[0].PortBegin, val.Rules[0].PortEnd)
	}
	for i := 1; i < MaxRulesPerIP; i++ {
		if val.Rules[i].IsValid() {
			t.Fatalf("rule[%d] should be invalid", i)
		}
	}
}

func TestNewValWithRules(t *testing.T) {
	rules := []CrapValRule{
		{PodIp: netip.MustParseAddr("10.0.0.1"), PortBegin: 80, PortEnd: 80},
		{PodIp: netip.MustParseAddr("10.0.0.2"), PortBegin: 443, PortEnd: 443},
		{PodIp: netip.MustParseAddr("10.0.0.3"), PortBegin: 3000, PortEnd: 4000},
	}

	val := NewValWithRules(rules)

	for i, want := range rules {
		if !val.Rules[i].IsValid() {
			t.Fatalf("rule[%d] should be valid", i)
		}
		if got := val.Rules[i].PodIp.Addr(); got != want.PodIp {
			t.Errorf("rule[%d] PodIp = %s, want %s", i, got, want.PodIp)
		}
		if val.Rules[i].PortBegin != want.PortBegin || val.Rules[i].PortEnd != want.PortEnd {
			t.Errorf("rule[%d] ports = %d-%d, want %d-%d", i, val.Rules[i].PortBegin, val.Rules[i].PortEnd, want.PortBegin, want.PortEnd)
		}
	}
	for i := len(rules); i < MaxRulesPerIP; i++ {
		if val.Rules[i].IsValid() {
			t.Fatalf("rule[%d] should be invalid after input rules", i)
		}
	}
}

func TestNewValWithRulesCompactsIPv6Holes(t *testing.T) {
	rules := []CrapValRule{
		{PodIp: netip.MustParseAddr("10.0.0.1"), PortBegin: 80, PortEnd: 80},
		{PodIp: netip.MustParseAddr("::1"), PortBegin: 9999, PortEnd: 9999},
		{PodIp: netip.MustParseAddr("10.0.0.2"), PortBegin: 443, PortEnd: 443},
	}

	val := NewValWithRules(rules)

	if !val.Rules[0].IsValid() || val.Rules[0].PodIp.Addr() != rules[0].PodIp {
		t.Fatalf("rule[0] = %+v, want pod_ip=%s", val.Rules[0], rules[0].PodIp)
	}
	if !val.Rules[1].IsValid() || val.Rules[1].PodIp.Addr() != rules[2].PodIp {
		t.Fatalf("rule[1] = %+v, want pod_ip=%s", val.Rules[1], rules[2].PodIp)
	}
	for i := 2; i < MaxRulesPerIP; i++ {
		if val.Rules[i].IsValid() {
			t.Fatalf("rule[%d] should be invalid", i)
		}
	}
}

func TestCrapValStringNonEmptyOnly(t *testing.T) {
	val := NewValWithRules([]CrapValRule{
		{PodIp: netip.MustParseAddr("10.0.0.1"), PortBegin: 80, PortEnd: 80},
		{PodIp: netip.MustParseAddr("10.0.0.2"), PortBegin: 443, PortEnd: 443},
	})

	s := val.String()
	if len(s) == 0 {
		t.Fatal("String() should not be empty")
	}
}

func TestCrapKeyFromIPv6IsZero(t *testing.T) {
	ipv6 := netip.MustParseAddr("::1")
	key := NewKey(ipv6)
	if key.DestIP != [4]byte{} {
		t.Fatal("expected zero key for IPv6")
	}
}
