// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package builder

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/cilium/cilium/cilium-cli/connectivity/check"
	"github.com/cilium/cilium/cilium-cli/connectivity/tests"
	"github.com/cilium/cilium/cilium-cli/defaults"
	"github.com/cilium/cilium/cilium-cli/utils/features"
	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/watch"
)

func CtNatCleanupCheck() check.Scenario {
	return &ctNatCleanupCheck{
		ScenarioBase: check.NewScenarioBase(),
	}
}

type ctNatCleanupCheck struct {
	check.ScenarioBase
}

func (s *ctNatCleanupCheck) Name() string {
	return "ct-nat-cleanup-check"
}

func bpfEntityList(ctx context.Context, agentPod check.Pod, entity string) ([]string, error) {
	out, err := agentPod.K8sClient.ExecInPod(
		ctx,
		agentPod.Namespace(),
		agentPod.NameWithoutNamespace(),
		defaults.AgentContainerName,
		[]string{
			"cilium",
			"bpf",
			entity,
			"list",
		})

	if err != nil {
		return nil, err
	}

	return strings.Split(out.String(), "\n"), nil
}

func ctList(ctx context.Context, agentPod check.Pod) ([]string, error) {
	return bpfEntityList(ctx, agentPod, "ct")
}

func natList(ctx context.Context, agentPod check.Pod) ([]string, error) {
	return bpfEntityList(ctx, agentPod, "nat")
}

func ctNatEntryFound(ctx context.Context, t *check.Test) bool {
	agents := t.Context().CiliumPods()
	echoPods := t.Context().EchoPods()
	var found *bool

	if len(agents) == 0 || len(echoPods) == 0 {
		return false
	}

	for _, agentPod := range agents {
		var ctEntryFound, natEntryFound bool

		ctlist, err := ctList(ctx, agentPod)

		if err != nil {
			t.Log(err)
		}

		natlist, err := natList(ctx, agentPod)

		if err != nil {
			t.Log(err)
		}

		for _, echoPod := range echoPods {
			addr := echoPod.Address(features.IPFamilyV4)

			for _, entry := range ctlist {
				if strings.Contains(entry, addr) {
					ctEntryFound = true
				}
			}

			for _, entry := range natlist {
				if strings.Contains(entry, addr) {
					natEntryFound = true
				}
			}
		}

		if found == nil {
			found = new(ctEntryFound && natEntryFound)
		} else {
			*found = *found && (ctEntryFound && natEntryFound)
		}
	}

	if found == nil {
		return false
	}

	return *found
}

func (s *ctNatCleanupCheck) generateTraffic(ctx context.Context, t *check.Test) {
	clientPod := t.Context().HostNetNSPodsByNode()[t.NodesWithoutCilium()[0]]
	i := 0

	// With kube-proxy doing N/S LB it is not possible to see the original client
	// IP, as iptables rules do the LB SNAT/DNAT before the packet hits any
	// of Cilium's datapath BPF progs. So, skip the flow validation in that case.
	status, ok := t.Context().Feature(features.KPR)
	validateFlows := ok && status.Enabled

	for _, svc := range t.Context().EchoServices() {
		for _, node := range t.Context().Nodes() {
			tests.CurlNodePort(ctx, s, t, fmt.Sprintf("curl-%d", i), &clientPod, svc, node, validateFlows, t.Context().Params().SecondaryNetworkIface != "")
			i++
		}
	}
}

func waitPods(
	ctx context.Context,
	t *check.Test,
	echoPods []check.Pod,
	stopCondition func(et watch.EventType, pod *corev1.Pod) bool,
) {
	var wg sync.WaitGroup

	for _, pod := range echoPods {
		wg.Go(func() {
			watcher, err := t.Context().K8sClient().Clientset.CoreV1().Pods(pod.Namespace()).Watch(ctx, v1.ListOptions{
				FieldSelector: fields.OneTermEqualSelector("metadata.name", pod.NameWithoutNamespace()).String(),
			})

			if err != nil {
				t.Fail(err)
				return
			}
			defer watcher.Stop()

		loop:
			for event := range watcher.ResultChan() {
				pod, ok := event.Object.(*corev1.Pod)
				if !ok {
					continue
				}

				switch event.Type {
				case watch.Added:
					t.Log(fmt.Sprintf("[%s] Pod added: phase=%s", pod.Name, pod.Status.Phase))
				case watch.Modified:
					t.Log(fmt.Sprintf("[%s] Pod modified: phase=%s", pod.Name, pod.Status.Phase))
				case watch.Deleted:
					t.Log(fmt.Sprintf("[%s] Pod deleted", pod.Name))
				case watch.Error:
					t.Log(fmt.Sprintf("[%s] Watch error: %v", pod.Name, event.Object))
				}

				if stopCondition(event.Type, pod) {
					break loop
				}
			}
		})
	}

	wg.Wait()
}

func echoPods(ctx context.Context, t *check.Test) []check.Pod {
	svcPods := []check.Pod{}

	for _, svc := range t.Context().EchoServices() {
		name := svc.Service.Spec.Selector["name"]

		pods, err := t.Context().K8sClient().ListPods(ctx, svc.Service.Namespace, v1.ListOptions{
			LabelSelector: fmt.Sprintf("name=%s", name),
		})

		if err != nil {
			t.Fail(err)
			continue
		}

		if len(pods.Items) > 0 {
			svcPods = append(svcPods, check.Pod{Pod: &pods.Items[0]})
		}
	}

	return svcPods
}

func (s *ctNatCleanupCheck) Run(ctx context.Context, t *check.Test) {
	s.generateTraffic(ctx, t)

	if !ctNatEntryFound(ctx, t) {
		t.Fatal("missing CT/NAT entries")
	}

	deletedPods := echoPods(ctx, t)

	if len(deletedPods) == 0 {
		t.Fatal("missing intial echo pods")
	}

	var wg sync.WaitGroup

	wg.Go(func() {
		waitPods(ctx, t, deletedPods, func(et watch.EventType, _ *corev1.Pod) bool {
			return et == watch.Deleted
		})
	})

	for _, echoPod := range deletedPods {
		if err := t.Context().K8sClient().DeletePod(
			ctx,
			echoPod.Namespace(),
			echoPod.NameWithoutNamespace(),
			v1.DeleteOptions{},
		); err != nil {
			t.Fail(err)
		}
	}

	wg.Wait()

	reCreatedPods := echoPods(ctx, t)

	if len(reCreatedPods) == 0 {
		t.Fatal("missing re-created echo pods")
	}

	waitPods(ctx, t, reCreatedPods, func(et watch.EventType, pod *corev1.Pod) bool {
		return et == watch.Modified && slices.ContainsFunc(pod.Status.Conditions, func(cond corev1.PodCondition) bool {
			return cond.Status == corev1.ConditionTrue && cond.Type == corev1.PodReady
		})
	})

	retries := 5

	for i := range retries {
		if !ctNatEntryFound(ctx, t) {
			return
		}

		t.Log(fmt.Sprintf("check stale CT/NAT entries try #%d", i+1))

		time.Sleep(time.Second)
	}

	t.Fail("stale CT/NAT entries")
}

type ctNatCleanup struct{}

func (t ctNatCleanup) build(ct *check.ConnectivityTest, _ map[string]string) {
	newTest("ct-nat-cleanup", ct).
		WithFeatureRequirements(
			withKPRReqForMultiCluster(ct, features.RequireEnabled(features.NodeWithoutCilium))...,
		).
		WithScenarios(
			CtNatCleanupCheck(),
		)
}
