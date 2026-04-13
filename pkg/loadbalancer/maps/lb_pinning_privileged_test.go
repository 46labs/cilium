// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package maps

import (
	"net"
	"testing"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/rlimit"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/cilium/cilium/pkg/bpf"
	"github.com/cilium/cilium/pkg/testutils"
	"github.com/cilium/cilium/pkg/types"
)

func TestPrivilegedPinningMap(t *testing.T) {
	testutils.PrivilegedTest(t)

	require.NoError(t, rlimit.RemoveMemlock())

	m := bpf.NewMap(
		"cilium_lb4_pinning_test",
		ebpf.Hash,
		&LbPinning4Key{},
		&LbPinning4Value{},
		10,
		0,
	)
	require.NoError(t, m.OpenOrCreate())
	t.Cleanup(func() {
		m.UnpinIfExists()
		m.Close()
	})

	svcIP := types.IPv4(net.ParseIP("10.0.0.1").To4())
	nodeIP := types.IPv4(net.ParseIP("192.168.1.1").To4())
	key := &LbPinning4Key{ServiceIP: svcIP}
	val := &LbPinning4Value{NodeIP: nodeIP}

	require.NoError(t, m.Update(key, val))

	got, err := m.Lookup(key)
	require.NoError(t, err)
	assert.Equal(t, val, got.(*LbPinning4Value))

	require.NoError(t, m.Delete(key))

	_, err = m.Lookup(key)
	assert.ErrorIs(t, err, ebpf.ErrKeyNotExist)
}

func TestPrivilegedPinningMapDump(t *testing.T) {
	testutils.PrivilegedTest(t)

	require.NoError(t, rlimit.RemoveMemlock())

	m := bpf.NewMap(
		"cilium_lb4_pinning_dump_test",
		ebpf.Hash,
		&LbPinning4Key{},
		&LbPinning4Value{},
		10,
		0,
	)
	require.NoError(t, m.OpenOrCreate())
	t.Cleanup(func() {
		m.UnpinIfExists()
		m.Close()
	})

	svcIP := types.IPv4(net.ParseIP("10.0.0.1").To4())
	nodeIP := types.IPv4(net.ParseIP("192.168.1.1").To4())
	key := &LbPinning4Key{ServiceIP: svcIP}
	val := &LbPinning4Value{NodeIP: nodeIP}
	require.NoError(t, m.Update(key, val))

	count := 0
	require.NoError(t, m.DumpWithCallback(func(k bpf.MapKey, v bpf.MapValue) {
		count++
		assert.Equal(t, key, k.(*LbPinning4Key))
		assert.Equal(t, val, v.(*LbPinning4Value))
	}))
	assert.Equal(t, 1, count)
}
