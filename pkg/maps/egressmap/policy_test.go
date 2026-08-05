// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package egressmap

import (
	"net/netip"
	"testing"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/rlimit"
	"github.com/cilium/hive/hivetest"
	"github.com/stretchr/testify/assert"

	"github.com/cilium/cilium/pkg/bpf"
	"github.com/cilium/cilium/pkg/testutils"
)

func TestPrivilegedPolicyMap(t *testing.T) {
	testutils.PrivilegedTest(t)

	logger := hivetest.Logger(t)
	bpf.CheckOrMountFS(logger, "")
	assert.NoError(t, rlimit.RemoveMemlock())

	t.Run("IPv4 policies", func(t *testing.T) {
		egressPolicyMap := createPolicyMap4(hivetest.Lifecycle(t), nil, DefaultPolicyConfig, ebpf.PinNone)

		sourceIP1 := netip.MustParseAddr("1.1.1.1")
		sourceIP2 := netip.MustParseAddr("1.1.1.2")

		destCIDR1 := netip.MustParsePrefix("2.2.1.0/24")
		destCIDR2 := netip.MustParsePrefix("2.2.2.0/24")

		egressIP1 := netip.MustParseAddr("3.3.3.1")
		egressIP2 := netip.MustParseAddr("3.3.3.2")

		err := egressPolicyMap.Update(sourceIP1, destCIDR1, 0, false, egressIP1, egressIP1, 0, false)
		assert.NoError(t, err)

		err = egressPolicyMap.Update(sourceIP2, destCIDR2, 0, false, egressIP2, egressIP2, 0, false)
		assert.NoError(t, err)

		val, err := egressPolicyMap.Lookup(sourceIP1, destCIDR1, 0, false)
		assert.NoError(t, err)

		assert.Equal(t, val.EgressIP.Addr(), egressIP1)
		assert.Equal(t, val.GatewayIP.Addr(), egressIP1)

		val, err = egressPolicyMap.Lookup(sourceIP2, destCIDR2, 0, false)
		assert.NoError(t, err)

		assert.Equal(t, val.EgressIP.Addr(), egressIP2)
		assert.Equal(t, val.GatewayIP.Addr(), egressIP2)

		err = egressPolicyMap.Delete(sourceIP2, destCIDR2, 0, false)
		assert.NoError(t, err)

		val, err = egressPolicyMap.Lookup(sourceIP1, destCIDR1, 0, false)
		assert.NoError(t, err)

		assert.Equal(t, val.EgressIP.Addr(), egressIP1)
		assert.Equal(t, val.GatewayIP.Addr(), egressIP1)

		_, err = egressPolicyMap.Lookup(sourceIP2, destCIDR2, 0, false)
		assert.ErrorIs(t, err, ebpf.ErrKeyNotExist)
	})

	t.Run("IPv6 policies", func(t *testing.T) {
		egressPolicyMap := createPolicyMap6(hivetest.Lifecycle(t), nil, DefaultPolicyConfig, ebpf.PinNone)

		sourceIP1 := netip.MustParseAddr("2001:db8:1::1")
		sourceIP2 := netip.MustParseAddr("2001:db8:1::2")

		destCIDR1 := netip.MustParsePrefix("2001:db8:2::/64")
		destCIDR2 := netip.MustParsePrefix("2001:db8:3::/64")

		egressIP1 := netip.MustParseAddr("2001:db8:4::1")
		egressIP2 := netip.MustParseAddr("2001:db8:4::2")

		gatewayIP1 := netip.MustParseAddr("3.3.3.1")
		gatewayIP2 := netip.MustParseAddr("3.3.3.2")

		ifIndex1 := uint32(1)
		ifIndex2 := uint32(2)

		err := egressPolicyMap.Update(sourceIP1, destCIDR1, egressIP1, gatewayIP1, ifIndex1)
		assert.NoError(t, err)

		err = egressPolicyMap.Update(sourceIP2, destCIDR2, egressIP2, gatewayIP2, ifIndex2)
		assert.NoError(t, err)

		val, err := egressPolicyMap.Lookup(sourceIP1, destCIDR1)
		assert.NoError(t, err)

		assert.Equal(t, val.EgressIP.Addr(), egressIP1)
		assert.Equal(t, val.GatewayIP.Addr(), gatewayIP1)
		assert.Equal(t, val.EgressIfindex, ifIndex1)

		val, err = egressPolicyMap.Lookup(sourceIP2, destCIDR2)
		assert.NoError(t, err)

		assert.Equal(t, val.EgressIP.Addr(), egressIP2)
		assert.Equal(t, val.GatewayIP.Addr(), gatewayIP2)
		assert.Equal(t, val.EgressIfindex, ifIndex2)

		err = egressPolicyMap.Delete(sourceIP2, destCIDR2)
		assert.NoError(t, err)

		val, err = egressPolicyMap.Lookup(sourceIP1, destCIDR1)
		assert.NoError(t, err)

		assert.Equal(t, val.EgressIP.Addr(), egressIP1)
		assert.Equal(t, val.GatewayIP.Addr(), gatewayIP1)
		assert.Equal(t, val.EgressIfindex, ifIndex1)

		_, err = egressPolicyMap.Lookup(sourceIP2, destCIDR2)
		assert.ErrorIs(t, err, ebpf.ErrKeyNotExist)
	})

	t.Run("IPv4 policies with pinned TOS", func(t *testing.T) {
		egressPolicyMap := createPolicyMap4(hivetest.Lifecycle(t), nil, DefaultPolicyConfig, ebpf.PinNone)

		sourceIP := netip.MustParseAddr("1.1.1.1")
		destCIDR := netip.MustParsePrefix("2.2.1.5/32")
		egressIP := netip.MustParseAddr("3.3.3.1")
		const tos = 0xb8

		// Fallback entry with a pinned TOS of 0 plus the TOS-pinned entry.
		err := egressPolicyMap.Update(sourceIP, destCIDR, 0, true, egressIP, egressIP, 0, false)
		assert.NoError(t, err)

		err = egressPolicyMap.Update(sourceIP, destCIDR, tos, true, egressIP, egressIP, 0, false)
		assert.NoError(t, err)

		val, err := egressPolicyMap.Lookup(sourceIP, destCIDR, tos, true)
		assert.NoError(t, err)
		assert.Equal(t, val.EgressIP.Addr(), egressIP)

		val, err = egressPolicyMap.Lookup(sourceIP, destCIDR, 0, true)
		assert.NoError(t, err)
		assert.Equal(t, val.EgressIP.Addr(), egressIP)

		// A lookup with a non-matching TOS falls back to the entry with a
		// pinned TOS of 0, mirroring the datapath's two-lookup semantics.
		val, err = egressPolicyMap.Lookup(sourceIP, destCIDR, tos+1, true)
		assert.NoError(t, err)
		assert.Equal(t, val.EgressIP.Addr(), egressIP)

		err = egressPolicyMap.Delete(sourceIP, destCIDR, tos, true)
		assert.NoError(t, err)

		// The pinned entry is gone; the lookup with the pinned TOS now falls
		// back to the entry with a pinned TOS of 0.
		val, err = egressPolicyMap.Lookup(sourceIP, destCIDR, tos, true)
		assert.NoError(t, err)
		assert.Equal(t, val.EgressIP.Addr(), egressIP)

		err = egressPolicyMap.Delete(sourceIP, destCIDR, 0, true)
		assert.NoError(t, err)

		_, err = egressPolicyMap.Lookup(sourceIP, destCIDR, tos, true)
		assert.ErrorIs(t, err, ebpf.ErrKeyNotExist)

		_, err = egressPolicyMap.Lookup(sourceIP, destCIDR, 0, true)
		assert.ErrorIs(t, err, ebpf.ErrKeyNotExist)
	})
}
