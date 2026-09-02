// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package reflectors

import (
	"net/netip"
	"testing"

	"github.com/cilium/hive/hivetest"
	"github.com/stretchr/testify/require"

	"github.com/cilium/cilium/pkg/annotation"
	"github.com/cilium/cilium/pkg/k8s/client/testutils"
	slim_corev1 "github.com/cilium/cilium/pkg/k8s/slim/k8s/api/core/v1"
	slim_metav1 "github.com/cilium/cilium/pkg/k8s/slim/k8s/apis/meta/v1"
)

func newTestPod(annotations map[string]string, podIP string) *slim_corev1.Pod {
	return &slim_corev1.Pod{
		ObjectMeta: slim_metav1.ObjectMeta{
			UID:         "test-uid",
			Namespace:   "test-ns",
			Name:        "test-pod",
			Annotations: annotations,
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

	t.Run("pod with source ranges annotation is transformed", func(t *testing.T) {
		pod := newTestPod(map[string]string{annotation.PodSourceRanges: "10.0.0.0/8"}, "10.1.0.1")
		got, ok := transform(nil, pod)
		require.True(t, ok, "expected pod with annotation and IP to be accepted")
		require.Equal(t, LbSrcRangeGroupPod{
			UID:          "test-uid",
			Namespace:    "test-ns",
			Name:         "test-pod",
			IP:           netip.MustParseAddr("10.1.0.1"),
			SourceRanges: "10.0.0.0/8",
		}, got)
	})

	t.Run("pod without the annotation is still transformed with empty source ranges", func(t *testing.T) {
		// This matters for the case where the PodSourceRanges annotation is
		// removed from an otherwise still-selected pod: the reflector must
		// keep reporting the pod (as an update, not a deletion) so that the
		// downstream observer can clear any previously-applied source ranges
		// on the matching backend.
		pod := newTestPod(nil, "10.1.0.1")
		got, ok := transform(nil, pod)
		require.True(t, ok, "pod without PodSourceRanges annotation must still be transformed")
		require.Equal(t, LbSrcRangeGroupPod{
			UID:       "test-uid",
			Namespace: "test-ns",
			Name:      "test-pod",
			IP:        netip.MustParseAddr("10.1.0.1"),
		}, got)
	})

	t.Run("pod without a pod IP is skipped", func(t *testing.T) {
		pod := newTestPod(map[string]string{annotation.PodSourceRanges: "10.0.0.0/8"}, "")
		_, ok := transform(nil, pod)
		require.False(t, ok, "pod without a parseable PodIP must be skipped")
	})

	t.Run("non-pod object is skipped", func(t *testing.T) {
		_, ok := transform(nil, &slim_corev1.Service{})
		require.False(t, ok, "non-Pod objects must be skipped")
	})
}
