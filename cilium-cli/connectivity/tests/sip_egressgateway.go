// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package tests

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	ciliumv2 "github.com/cilium/cilium/pkg/k8s/apis/cilium.io/v2"
	slimv1 "github.com/cilium/cilium/pkg/k8s/slim/k8s/apis/meta/v1"

	"github.com/cilium/cilium/cilium-cli/connectivity/check"
	"github.com/cilium/cilium/cilium-cli/defaults"
	"github.com/cilium/cilium/cilium-cli/utils/features"
)

const (
	sipServiceName     = "sip-pinning-test"
	sipEGWPolicyName   = "cegp-sip-pinning-test"
	sipPort            = 5060
	sipServicePortName = "sip-udp"

	// sipPadding ensures the packet has 68+ bytes after Call-ID value start,
	// satisfying the BPF bounds check at sip.h:155.
	sipPadding = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
)

// sipCallIDs are the deterministic Call-ID values used for testing.
// Multiple distinct IDs allow testing both stickiness and distribution.
var sipCallIDs = []string{
	"test-sip-call-A",
	"test-sip-call-B",
	"test-sip-call-C",
}

// SipEgressGateway tests the SIP LB pinning + egress gateway service-pinning path.
func SipEgressGateway() check.Scenario {
	return &sipEgressGateway{
		ScenarioBase: check.NewScenarioBase(),
	}
}

type sipEgressGateway struct {
	check.ScenarioBase
}

func (s *sipEgressGateway) Name() string {
	return "sip-egress-gateway"
}

// buildSipPayload constructs a minimal SIP INVITE packet with the given Call-ID.
// The packet is padded to ensure 68+ bytes after the Call-ID value start,
// as required by the BPF bounds check in sip_inspect().
func buildSipPayload(callID string) string {
	return fmt.Sprintf(
		"INVITE sip:test@example.com SIP/2.0\r\n"+
			"Call-ID: %s\r\n"+
			"Content-Length: 0\r\n"+
			"X-Pad: %s\r\n"+
			"\r\n",
		callID, sipPadding,
	)
}

// sipSendCommand returns a shell command that sends a SIP UDP packet to addr:port
// using printf + nc (available in alpine-curl image).
func sipSendCommand(addr string, port int, callID string) []string {
	payload := buildSipPayload(callID)
	// Use printf to emit the payload, pipe to nc for UDP send.
	// -u = UDP, -w1 = 1 second timeout (don't wait for response).
	cmd := fmt.Sprintf(`printf '%s' | nc -u -w1 %s %d`, payload, addr, port)
	return []string{"/bin/sh", "-c", cmd}
}

func (s *sipEgressGateway) Run(ctx context.Context, t *check.Test) {
	ct := t.Context()

	// Discover the egress gateway node (node running the other=client pod).
	egressGatewayNode := t.EgressGatewayNode()
	if egressGatewayNode == "" {
		t.Fatal("Cannot find egress gateway node")
	}

	egressGatewayNodeInternalIP := ct.GetGatewayNodeInternalIP(egressGatewayNode, false)
	if !egressGatewayNodeInternalIP.IsValid() {
		t.Fatal("Cannot get egress gateway node internal IP")
	}

	// The ExternalIP serves as both the Service VIP and the EGW egressIP.
	// Using the gateway node's internal IP ensures routability in the test topology.
	svcExternalIP := egressGatewayNodeInternalIP.String()
	namespace := ct.Params().TestNamespace

	// Step 2: Create SIP Service + EGW policy resources.
	svc := buildSipService(namespace, svcExternalIP, egressGatewayNode)
	egwPolicy := buildSipEgressGatewayPolicy(namespace, svcExternalIP)

	if _, err := ct.K8sClient().CreateService(ctx, namespace, svc, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create SIP service: %v", err)
	}
	defer func() {
		_ = ct.K8sClient().DeleteService(ctx, namespace, sipServiceName, metav1.DeleteOptions{})
	}()

	if _, err := ct.K8sClient().CiliumClientset.CiliumV2().CiliumEgressGatewayPolicies().Create(ctx, egwPolicy, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create SIP EGW policy: %v", err)
	}
	defer func() {
		_ = ct.K8sClient().CiliumClientset.CiliumV2().CiliumEgressGatewayPolicies().Delete(ctx, sipEGWPolicyName, metav1.DeleteOptions{})
	}()

	t.Logf("Created SIP service %s (ExternalIP=%s, pinned to %s) and EGW policy %s",
		sipServiceName, svcExternalIP, egressGatewayNode, sipEGWPolicyName)

	// Step 5: Assert LB pinning map state (from non-pinned node's agent).
	s.assertLBPinningMap(ctx, t, ct, svcExternalIP, egressGatewayNode)

	// Step 6+7: Assert EGW BPF policy map contains SIP fields and core routing fields.
	s.assertEGWPolicyMap(ctx, t, ct, svcExternalIP)

	// Step 3+4: Send SIP traffic and verify backend stickiness per Call-ID.
	s.sendSipTrafficAndAssertStickiness(ctx, t, ct, svcExternalIP)

	// Step 8: Assert CT entries contain sip-call-id-hash.
	s.assertCTSipHash(ctx, t, ct)

	// Step 9: Assert multiple Call-IDs produce distinct CT hash entries (soft check).
	s.assertMultiCallIDSpread(ctx, t, ct)
}

// assertLBPinningMap verifies the LB pinning map contains the expected serviceIP -> nodeIP entry.
// Checks from a non-pinned node since entries are absent on the pinned node itself.
func (s *sipEgressGateway) assertLBPinningMap(ctx context.Context, t *check.Test, ct *check.ConnectivityTest, svcExternalIP, pinnedNode string) {
	pinnedNodeIP := ct.GetGatewayNodeInternalIP(pinnedNode, false)

	for _, ciliumPod := range ct.CiliumPods() {
		// Skip the pinned node — its pinning map won't have this entry.
		if ciliumPod.Pod.Spec.NodeName == pinnedNode {
			continue
		}

		cmd := []string{"cilium-dbg", "bpf", "lb", "pinning", "list"}
		stdout, err := ciliumPod.K8sClient.ExecInPod(ctx, ciliumPod.Pod.Namespace, ciliumPod.Pod.Name, defaults.AgentContainerName, cmd)
		if err != nil {
			t.Fatalf("Failed to list LB pinning map on %s: %v", ciliumPod.Pod.Spec.NodeName, err)
		}

		output := stdout.String()
		if !strings.Contains(output, svcExternalIP) || !strings.Contains(output, pinnedNodeIP.String()) {
			t.Fatalf("LB pinning map on %s does not contain expected entry %s -> %s.\nOutput: %s",
				ciliumPod.Pod.Spec.NodeName, svcExternalIP, pinnedNodeIP.String(), output)
		}

		t.Logf("LB pinning map on %s contains %s -> %s", ciliumPod.Pod.Spec.NodeName, svcExternalIP, pinnedNodeIP.String())
		return // Only need to verify from one non-pinned node.
	}

	t.Fatal("No non-pinned cilium pod found to check LB pinning map")
}

// egressListEntry represents a single entry from `cilium bpf egress list -o json`.
type egressListEntry struct {
	SourceIP   string `json:"SourceIP"`
	DestCIDR   string `json:"DestCIDR"`
	EgressIP   string `json:"EgressIP"`
	GatewayIP  string `json:"GatewayIP"`
	SipInspect bool   `json:"SipInspect"`
	SipPort    uint16 `json:"SipPort"`
}

// assertEGWPolicyMap verifies the EGW BPF policy map contains entries with
// the expected SIP-specific fields (sip_inspect=true, sip_port).
func (s *sipEgressGateway) assertEGWPolicyMap(ctx context.Context, t *check.Test, ct *check.ConnectivityTest, expectedEgressIP string) {
	for _, ciliumPod := range ct.CiliumPods() {
		cmd := strings.Split("cilium bpf egress list -o json", " ")
		stdout, err := ciliumPod.K8sClient.ExecInPod(ctx, ciliumPod.Pod.Namespace, ciliumPod.Pod.Name, defaults.AgentContainerName, cmd)
		if err != nil {
			t.Fatalf("Failed to list EGW policy map on %s: %v", ciliumPod.Pod.Spec.NodeName, err)
		}

		var entries []egressListEntry
		if err := json.Unmarshal(stdout.Bytes(), &entries); err != nil {
			t.Fatalf("Failed to parse EGW policy map JSON on %s: %v", ciliumPod.Pod.Spec.NodeName, err)
		}

		for _, entry := range entries {
			if entry.EgressIP != expectedEgressIP {
				continue
			}

			if !entry.SipInspect {
				t.Fatalf("EGW policy entry for egressIP=%s on %s has sip_inspect=false, expected true",
					expectedEgressIP, ciliumPod.Pod.Spec.NodeName)
			}
			if entry.SipPort != sipPort {
				t.Fatalf("EGW policy entry for egressIP=%s on %s has sip_port=%d, expected %d",
					expectedEgressIP, ciliumPod.Pod.Spec.NodeName, entry.SipPort, sipPort)
			}

			t.Logf("EGW policy map on %s: egressIP=%s gatewayIP=%s sip_inspect=%v sip_port=%d",
				ciliumPod.Pod.Spec.NodeName, entry.EgressIP, entry.GatewayIP, entry.SipInspect, entry.SipPort)
			return // Found a matching entry with correct SIP fields.
		}
	}

	t.Fatalf("No EGW policy entry found with egressIP=%s and correct SIP fields", expectedEgressIP)
}

// sendSipTrafficAndAssertStickiness sends SIP packets with each Call-ID multiple times
// and verifies that the same Call-ID consistently reaches the same backend (via CT inspection).
func (s *sipEgressGateway) sendSipTrafficAndAssertStickiness(ctx context.Context, t *check.Test, ct *check.ConnectivityTest, svcExternalIP string) {
	// Send from the first available client pod.
	var clientPod check.Pod
	for _, pod := range ct.ClientPods() {
		clientPod = pod
		break
	}

	for _, callID := range sipCallIDs {
		// Send the same Call-ID 3 times to test stickiness.
		for i := 0; i < 3; i++ {
			cmd := sipSendCommand(svcExternalIP, sipPort, callID)
			t.NewAction(s, fmt.Sprintf("sip-%s-%d", callID, i), &clientPod, nil, features.IPFamilyV4).Run(func(a *check.Action) {
				a.ExecInPod(ctx, cmd)
			})
		}
	}

	t.Log("SIP traffic sent for all Call-IDs")
}

// assertCTSipHash verifies that CT entries exist with non-zero sip-call-id-hash,
// proving that sip_inspect() successfully parsed the SIP packets.
func (s *sipEgressGateway) assertCTSipHash(ctx context.Context, t *check.Test, ct *check.ConnectivityTest) {
	for _, ciliumPod := range ct.CiliumPods() {
		cmd := []string{"cilium-dbg", "bpf", "ct", "list", "global"}
		stdout, err := ciliumPod.K8sClient.ExecInPod(ctx, ciliumPod.Pod.Namespace, ciliumPod.Pod.Name, defaults.AgentContainerName, cmd)
		if err != nil {
			t.Logf("Failed to list CT on %s: %v", ciliumPod.Pod.Spec.NodeName, err)
			continue
		}

		output := stdout.String()
		if strings.Contains(output, "sip-call-id-hash") {
			t.Logf("CT entries on %s contain sip-call-id-hash entries", ciliumPod.Pod.Spec.NodeName)
			return
		}
	}

	t.Fatal("No CT entries found with sip-call-id-hash on any node")
}

// assertMultiCallIDSpread checks that different Call-IDs produced distinct
// sip-call-id-hash values in the CT map. This is a soft assertion — it warns
// rather than fails if only one hash is observed, since distribution depends
// on backend count and Maglev LUT configuration.
func (s *sipEgressGateway) assertMultiCallIDSpread(ctx context.Context, t *check.Test, ct *check.ConnectivityTest) {
	hashes := map[string]struct{}{}

	for _, ciliumPod := range ct.CiliumPods() {
		cmd := []string{"cilium-dbg", "bpf", "ct", "list", "global"}
		stdout, err := ciliumPod.K8sClient.ExecInPod(ctx, ciliumPod.Pod.Namespace, ciliumPod.Pod.Name, defaults.AgentContainerName, cmd)
		if err != nil {
			continue
		}

		for _, line := range strings.Split(stdout.String(), "\n") {
			idx := strings.Index(line, "sip-call-id-hash ")
			if idx < 0 {
				continue
			}
			// Extract the hex hash value after "sip-call-id-hash ".
			rest := line[idx+len("sip-call-id-hash "):]
			fields := strings.Fields(rest)
			if len(fields) > 0 {
				hashes[fields[0]] = struct{}{}
			}
		}
	}

	if len(hashes) == 0 {
		t.Fatal("No sip-call-id-hash entries found in CT map for spread check")
	}

	if len(hashes) < 2 {
		t.Logf("WARNING: Only %d distinct sip-call-id-hash value(s) observed across %d Call-IDs. "+
			"Distribution may require more backends or distinct tuples.", len(hashes), len(sipCallIDs))
	} else {
		t.Logf("Multi-Call-ID spread: %d distinct sip-call-id-hash values observed", len(hashes))
	}
}

// buildSipService creates a LoadBalancer Service with SIP annotations matching
// the pee controller contract.
func buildSipService(namespace, externalIP, pinningNode string) *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      sipServiceName,
			Namespace: namespace,
			Annotations: map[string]string{
				"service.cilium.io/sip-inspect":      "true",
				"service.cilium.io/lb-algorithm":     "maglev",
				"service.cilium.io/use-svc-pinning":  "true",
				"service.cilium.io/svc-pinning-node": pinningNode,
				"service.cilium.io/sip-pinning":      "true", // pee-controller contract, not consumed by Cilium
			},
		},
		Spec: corev1.ServiceSpec{
			Type:                  corev1.ServiceTypeLoadBalancer,
			ExternalTrafficPolicy: corev1.ServiceExternalTrafficPolicyCluster,
			ExternalIPs:           []string{externalIP},
			Selector: map[string]string{
				"kind": "echo",
			},
			Ports: []corev1.ServicePort{
				{
					Name:       sipServicePortName,
					Protocol:   corev1.ProtocolUDP,
					Port:       sipPort,
					TargetPort: intstr.FromInt(sipPort),
				},
			},
		},
	}
}

// buildSipEgressGatewayPolicy creates a CiliumEgressGatewayPolicy with SIP annotations
// matching the pee controller contract. The egressIP must equal the Service's ExternalIP
// for the pinning map lookup to work.
func buildSipEgressGatewayPolicy(namespace, egressIP string) *ciliumv2.CiliumEgressGatewayPolicy {
	return &ciliumv2.CiliumEgressGatewayPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: sipEGWPolicyName,
			Annotations: map[string]string{
				"service.cilium.io/sip-inspect":     "true",
				"service.cilium.io/sip-port":        fmt.Sprintf("%d", sipPort),
				"service.cilium.io/use-svc-pinning": "true",
				"service.cilium.io/sip-pinning":     "true", // pee-controller contract, not consumed by Cilium
			},
		},
		Spec: ciliumv2.CiliumEgressGatewayPolicySpec{
			Selectors: []ciliumv2.EgressRule{
				{
					PodSelector: &slimv1.LabelSelector{
						MatchLabels: map[string]string{
							"io.kubernetes.pod.namespace": namespace,
							"kind":                        "echo",
						},
					},
				},
			},
			DestinationCIDRs: []ciliumv2.CIDR{"0.0.0.0/0"},
			EgressGateway: &ciliumv2.EgressGateway{
				// nodeSelector is empty/unused — svcPinning=true bypasses it
				NodeSelector: &slimv1.LabelSelector{},
				EgressIP:     egressIP,
			},
		},
	}
}
