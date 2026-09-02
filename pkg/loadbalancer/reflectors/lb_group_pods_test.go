// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package reflectors

import (
	"net/netip"
	"testing"

	"github.com/cilium/hive/hivetest"
	"github.com/stretchr/testify/require"

	"github.com/cilium/cilium/pkg/k8s/client/testutils"
	slim_corev1 "github.com/cilium/cilium/pkg/k8s/slim/k8s/api/core/v1"
	slim_metav1 "github.com/cilium/cilium/pkg/k8s/slim/k8s/apis/meta/v1"
)

func newTestPod(labels map[string]string, podIP string) *slim_corev1.Pod {
	return &slim_corev1.Pod{
		ObjectMeta: slim_metav1.ObjectMeta{
			UID:       "test-uid",
			Namespace: "test-ns",
			Name:      "test-pod",
			Labels:    labels,
		},
		Status: slim_corev1.PodStatus{
			PodIP: podIP,
		},
	}
}

func TestPodReflectorConfigTransform(t *testing.T) {
	_, cs := testutils.NewFakeClientset(hivetest.Logger(t))
	cfg := podReflectorConfig(cs, nil)
	transform := cfg.Transform
	require.NotNil(t, transform, "Transform must be set")

	t.Run("valid pod is transformed", func(t *testing.T) {
		pod := newTestPod(map[string]string{PodSourceRangeGroup: "2"}, "10.1.0.1")
		got, ok := transform(nil, pod)
		require.True(t, ok, "expected pod with label and IP to be accepted")
		require.Equal(t, LbSrcRangeGroupPod{
			UID:        "test-uid",
			Namespace:  "test-ns",
			Name:       "test-pod",
			IP:         netip.MustParseAddr("10.1.0.1"),
			GroupIndex: 2,
		}, got)
	})

	t.Run("pod without the label is skipped", func(t *testing.T) {
		pod := newTestPod(nil, "10.1.0.1")
		_, ok := transform(nil, pod)
		require.False(t, ok, "pod without PodSourceRangeGroup label must be skipped")
	})

	t.Run("pod with non-numeric group value is skipped", func(t *testing.T) {
		pod := newTestPod(map[string]string{PodSourceRangeGroup: "not-a-number"}, "10.1.0.1")
		_, ok := transform(nil, pod)
		require.False(t, ok, "pod with an invalid group index must be skipped")
	})

	t.Run("pod with group value out of uint8 range is skipped", func(t *testing.T) {
		pod := newTestPod(map[string]string{PodSourceRangeGroup: "256"}, "10.1.0.1")
		_, ok := transform(nil, pod)
		require.False(t, ok, "pod with a group index above 255 must be skipped")
	})

	t.Run("pod without a pod IP is skipped", func(t *testing.T) {
		pod := newTestPod(map[string]string{PodSourceRangeGroup: "0"}, "")
		_, ok := transform(nil, pod)
		require.False(t, ok, "pod without a parseable PodIP must be skipped")
	})

	t.Run("non-pod object is skipped", func(t *testing.T) {
		_, ok := transform(nil, &slim_corev1.Service{})
		require.False(t, ok, "non-Pod objects must be skipped")
	})
}
