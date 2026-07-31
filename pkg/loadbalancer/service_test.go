// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package loadbalancer

import (
	"net/netip"
	"reflect"
	"strings"
	"testing"
)

func mustPrefix(t *testing.T, s string) netip.Prefix {
	t.Helper()
	p, err := netip.ParsePrefix(s)
	if err != nil {
		t.Fatalf("ParsePrefix(%q): %v", s, err)
	}
	return p.Masked()
}

func TestParseSourceRangeIndexes(t *testing.T) {
	tests := []struct {
		name    string
		value   string
		want    []SourceRangeIndexEntry
		wantErr string
	}{
		{
			name:  "empty value",
			value: "",
			want:  nil,
		},
		{
			name:  "multi-CIDR group shares index",
			value: "10.0.0.0/8, 10.1.0.0/16; 20.0.0.0/8:5060",
			want: []SourceRangeIndexEntry{
				{Index: 0, Prefix: mustPrefix(t, "10.0.0.0/8")},
				{Index: 0, Prefix: mustPrefix(t, "10.1.0.0/16")},
				{Index: 1, Prefix: mustPrefix(t, "20.0.0.0/8"), Port: 5060},
			},
		},
		{
			name:  "comma-only value is a single group with per-CIDR ports",
			value: "10.0.0.0/8,10.1.0.0/16:5060",
			want: []SourceRangeIndexEntry{
				{Index: 0, Prefix: mustPrefix(t, "10.0.0.0/8")},
				{Index: 0, Prefix: mustPrefix(t, "10.1.0.0/16"), Port: 5060},
			},
		},
		{
			name:  "bare IP defaults to full-length prefix",
			value: "192.168.1.1",
			want: []SourceRangeIndexEntry{
				{Index: 0, Prefix: mustPrefix(t, "192.168.1.1/32")},
			},
		},
		{
			name:  "IPv6 bracketed with port",
			value: "[fd00::1]/128:5060",
			want: []SourceRangeIndexEntry{
				{Index: 0, Prefix: mustPrefix(t, "fd00::1/128"), Port: 5060},
			},
		},
		{
			name:  "IPv6 unbracketed full CIDR",
			value: "fd00::1/128",
			want: []SourceRangeIndexEntry{
				{Index: 0, Prefix: mustPrefix(t, "fd00::1/128")},
			},
		},
		{
			name:  "empty groups are skipped",
			value: "10.0.0.0/8;;20.0.0.0/8",
			want: []SourceRangeIndexEntry{
				{Index: 0, Prefix: mustPrefix(t, "10.0.0.0/8")},
				{Index: 1, Prefix: mustPrefix(t, "20.0.0.0/8")},
			},
		},
		{
			name:    "invalid CIDR",
			value:   "not-a-cidr",
			wantErr: "invalid source range index",
		},
		{
			name:    "non-numeric suffix is not treated as a port",
			value:   "10.0.0.0/8:notaport",
			wantErr: "invalid source range index",
		},
		{
			name:    "out of range port is not treated as a port",
			value:   "10.0.0.0/8:99999",
			wantErr: "invalid source range index",
		},
		{
			name:    "invalid bracketed IPv6 port",
			value:   "[fd00::1]/128:notaport",
			wantErr: "invalid source range index port",
		},
		{
			name:    "unbalanced IPv6 bracket",
			value:   "[fd00::1/128",
			wantErr: "missing closing bracket",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseSourceRangeIndexes(tt.value)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("ParseSourceRangeIndexes(%q) error = %v, want containing %q", tt.value, err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseSourceRangeIndexes(%q) unexpected error: %v", tt.value, err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("ParseSourceRangeIndexes(%q) = %v, want %v", tt.value, got, tt.want)
			}
		})
	}
}

func TestParseSourceRangeIndexesTooMany(t *testing.T) {
	t.Run("too many groups", func(t *testing.T) {
		groups := make([]string, 258)
		for i := range groups {
			groups[i] = "10.0.0.0/8"
		}
		_, err := ParseSourceRangeIndexes(strings.Join(groups, ";"))
		if err == nil || !strings.Contains(err.Error(), "too many source range index groups") {
			t.Fatalf("expected group overflow error, got %v", err)
		}
	})
}
