// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package crap

import (
	"fmt"
	"net/netip"
	"strings"

	"log/slog"

	"golang.org/x/sys/unix"

	"github.com/cilium/cilium/pkg/bpf"
	"github.com/cilium/cilium/pkg/ebpf"
	"github.com/cilium/cilium/pkg/metrics"
	"github.com/cilium/cilium/pkg/option"
	"github.com/cilium/cilium/pkg/types"
	"github.com/cilium/hive/cell"
)

const (
	MaxEntries    = 8192
	MaxRulesPerIP = 8
	CrapMapName   = "cilium_crap_map"
)

// Must be in sync with struct crap_key in <bpf/lib/crap.h>
type CrapKey struct {
	DestIP types.IPv4 `align:"dst_ip"`
}

func (k CrapKey) String() string {
	return fmt.Sprintf("%s", k.DestIP)
}

func (k *CrapKey) New() bpf.MapKey { return &CrapKey{} }

func NewKey(dstIP netip.Addr) CrapKey {
	result := CrapKey{}

	if !dstIP.Is4() {
		return result
	}

	ip4 := dstIP.As4()
	copy(result.DestIP[:], ip4[:])

	return result
}

// Must be in sync with struct crap_rule in <bpf/lib/crap.h>
type CrapRule struct {
	PodIp     types.IPv4 `align:"pod_ip"`
	PortBegin uint16     `align:"port_begin"`
	PortEnd   uint16     `align:"port_end"`
}

func (r *CrapRule) String() string {
	return fmt.Sprintf("pod_ip=%s ports=%d-%d", r.PodIp, r.PortBegin, r.PortEnd)
}

func (r *CrapRule) IsValid() bool {
	return r.PodIp != [4]byte{}
}

// Must be in sync with struct crap_value in <bpf/lib/crap.h>
type CrapVal struct {
	Rules [MaxRulesPerIP]CrapRule `align:"rules"`
}

func (v *CrapVal) String() string {
	var sb strings.Builder
	for i := range v.Rules {
		if !v.Rules[i].IsValid() {
			break
		}
		if sb.Len() > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(v.Rules[i].String())
	}
	return sb.String()
}

func (v *CrapVal) New() bpf.MapValue { return &CrapVal{} }

func NewVal(podIp netip.Addr, portBegin, portEnd uint16) CrapVal {
	val := CrapVal{}
	if !podIp.Is4() {
		return val
	}
	ip4 := podIp.As4()
	copy(val.Rules[0].PodIp[:], ip4[:])
	val.Rules[0].PortBegin = portBegin
	val.Rules[0].PortEnd = portEnd
	return val
}

func NewValWithRules(rules []CrapValRule) CrapVal {
	val := CrapVal{}
	idx := 0
	for _, r := range rules {
		if idx >= MaxRulesPerIP {
			break
		}
		if !r.PodIp.Is4() {
			continue
		}
		ip4 := r.PodIp.As4()
		copy(val.Rules[idx].PodIp[:], ip4[:])
		val.Rules[idx].PortBegin = r.PortBegin
		val.Rules[idx].PortEnd = r.PortEnd
		idx++
	}
	return val
}

type CrapValRule struct {
	PodIp     netip.Addr
	PortBegin uint16
	PortEnd   uint16
}

// Map represents an CRAP BPF map.
type CrapMap struct {
	m *bpf.Map
}

func createPolicyMapFromDaemonConfig(lifecycle cell.Lifecycle, cfg *option.DaemonConfig, metricsRegistry *metrics.Registry) bpf.MapOut[*CrapMap] {
	if !cfg.EnableIPv4 {
		return bpf.NewMapOut[*CrapMap](nil)
	}

	return bpf.NewMapOut(newCrapMap(lifecycle, metricsRegistry, ebpf.PinByName))
}

// CreatePrivatePolicyMap creates an unpinned CRAP map.
//
// Useful for testing.
func CreatePrivatePolicyMap(lc cell.Lifecycle, registry *metrics.Registry) *CrapMap {
	return newCrapMap(lc, registry, ebpf.PinNone)
}

func newCrapMap(lc cell.Lifecycle, registry *metrics.Registry, pinning ebpf.PinType) *CrapMap {
	m := bpf.NewMap(
		CrapMapName,
		ebpf.Hash,
		&CrapKey{},
		&CrapVal{},
		MaxEntries,
		unix.BPF_F_NO_PREALLOC,
	).WithCache().WithPressureMetric(registry).
		WithEvents(option.Config.GetEventBufferConfig(CrapMapName))

	lc.Append(cell.Hook{
		OnStart: func(cell.HookContext) error {
			switch pinning {
			case ebpf.PinNone:
				return m.CreateUnpinned()
			case ebpf.PinByName:
				return m.OpenOrCreate()
			}
			return fmt.Errorf("received unexpected pin type: %d", pinning)
		},
		OnStop: func(cell.HookContext) error {
			return m.Close()
		},
	})

	return &CrapMap{m}
}

// OpenPinnedCrapMap opens an existing pinned CRAP map.
func OpenPinnedCrapMap(logger *slog.Logger) (*CrapMap, error) {
	m, err := bpf.OpenMap(bpf.MapPath(logger, CrapMapName), &CrapKey{}, &CrapVal{})
	if err != nil {
		return nil, err
	}

	return &CrapMap{m}, nil
}

func (m *CrapMap) UpdateCrapMapping(dstIP netip.Addr, podIp netip.Addr, portBegin, portEnd uint16) error {
	key := NewKey(dstIP)
	value := NewVal(podIp, portBegin, portEnd)

	return m.m.Update(&key, &value)
}

func (m *CrapMap) Update(key CrapKey, value CrapVal) error {
	return m.m.Update(&key, &value)
}

func (m *CrapMap) RemoveCrapMapping(dstIP netip.Addr) error {
	key := NewKey(dstIP)
	return m.m.Delete(&key)
}

func (m *CrapMap) Delete(key *CrapKey) error {
	return m.m.Delete(key)
}

func (m *CrapMap) Lookup(key *CrapKey) (*CrapVal, error) {
	ret, err := m.m.Lookup(key)
	if err != nil {
		return nil, err
	}
	return ret.(*CrapVal), err
}

// CrapIterateCallback represents the signature of the callback function
// expected by the IterateWithCallback method, which in turn is used to iterate
// all the keys/values of an crap bpf map.
type CrapIterateCallback func(*CrapKey, *CrapVal)

// IterateWithCallback iterates through all the keys/values of crap rules
// map, passing each key/value pair to the cb callback.
func (m *CrapMap) IterateWithCallback(cb CrapIterateCallback) error {
	return m.m.DumpWithCallback(func(k bpf.MapKey, v bpf.MapValue) {
		key := k.(*CrapKey)
		value := v.(*CrapVal)

		cb(key, value)
	})
}

func (k *CrapKey) Match(dst_ip netip.Addr) bool {
	nkey := NewKey(dst_ip)
	return nkey == *k
}
