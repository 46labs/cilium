// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package egressgateway

import (
	"net/netip"
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"github.com/cilium/hive/hivetest"
	"github.com/cilium/cilium/pkg/annotation"
	v2 "github.com/cilium/cilium/pkg/k8s/apis/cilium.io/v2"
	slimv1 "github.com/cilium/cilium/pkg/k8s/slim/k8s/apis/meta/v1"
	"github.com/cilium/cilium/pkg/labels"
	"github.com/cilium/cilium/pkg/policy/api"
	policyTypes "github.com/cilium/cilium/pkg/policy/types"
)

func getAsPolicyLabelSelectors(k8sLss []*slimv1.LabelSelector) (lss []*policyTypes.LabelSelector) {
	for _, ls := range k8sLss {
		lss = append(lss, policyTypes.NewLabelSelector(api.NewESFromK8sLabelSelector(labels.LabelSourceK8sKeyPrefix, ls)))
	}
	return lss
}

func TestPolicyConfig_updateMatchedEndpointIDs(t *testing.T) {
	type fields struct {
		id                types.NamespacedName
		endpointSelectors []*slimv1.LabelSelector
		nodeSelectors     []*slimv1.LabelSelector
		dstCIDRs          []netip.Prefix
		excludedCIDRs     []netip.Prefix
		policyGwConfigs   []policyGatewayConfig
		matchedEndpoints  map[endpointID]*endpointMetadata
		gatewayConfigs    []gatewayConfig
	}
	type args struct {
		epDataStore           map[endpointID]*endpointMetadata
		nodesAddresses2Labels map[string]map[string]string
	}
	tests := []struct {
		name           string
		fields         fields
		args           args
		want           int
		wantEndpointID endpointID
	}{
		{
			name: "Test updateMatchedEndpointIDs with endpoints and nodes",
			fields: fields{
				id: types.NamespacedName{
					Name: "test",
				},
				endpointSelectors: []*slimv1.LabelSelector{{
					MatchLabels: map[string]string{
						"app": "test",
					},
				}},
				nodeSelectors: []*slimv1.LabelSelector{{
					MatchLabels: map[string]string{
						"node-name": "node1",
					},
				}},
			},
			args: args{
				epDataStore: map[endpointID]*endpointMetadata{
					"123456": {
						id: "123456",
						labels: map[string]string{
							"app": "test",
						},
						nodeIP: "192.168.1.10",
					},
				},
				nodesAddresses2Labels: map[string]map[string]string{
					"192.168.1.10": {
						"node-name": "node1",
					},
				},
			},
			want:           1,
			wantEndpointID: endpointID("123456"),
		},
		{
			name: "Test updateMatchedEndpointIDs with namespaced endpoints and nodes",
			fields: fields{
				id: types.NamespacedName{
					Name: "test",
				},
				endpointSelectors: []*slimv1.LabelSelector{{
					MatchLabels: map[string]string{
						"io.kubernetes.pod.namespace": "default",
						"app":                         "test",
					},
				}},
				nodeSelectors: []*slimv1.LabelSelector{{
					MatchLabels: map[string]string{
						"node-name": "node1",
					},
				}},
			},
			args: args{
				epDataStore: map[endpointID]*endpointMetadata{
					"123456": {
						id: "123456",
						labels: map[string]string{
							"io.kubernetes.pod.namespace": "default",
							"app":                         "test",
						},
						nodeIP: "192.168.1.10",
					},
				},
				nodesAddresses2Labels: map[string]map[string]string{
					"192.168.1.10": {
						"node-name": "node1",
					},
				},
			},
			want:           1,
			wantEndpointID: endpointID("123456"),
		},
		{
			name: "Test updateMatchedEndpointIDs endpoints and nodes with no match",
			fields: fields{
				id: types.NamespacedName{
					Name: "test",
				},
				endpointSelectors: []*slimv1.LabelSelector{{
					MatchLabels: map[string]string{
						"app": "test",
					},
				}},
				nodeSelectors: []*slimv1.LabelSelector{{
					MatchLabels: map[string]string{
						"node-name": "node1",
					},
				}},
			},
			args: args{
				epDataStore: map[endpointID]*endpointMetadata{
					"123456": {
						id: "123456",
						labels: map[string]string{
							"app": "test",
						},
						nodeIP: "192.168.1.10",
					},
				},
				nodesAddresses2Labels: map[string]map[string]string{
					"192.168.1.11": {
						"node-name": "node1",
					},
				},
			},
			want:           0,
			wantEndpointID: "",
		},
		{
			name: "Test updateMatchedEndpointIDs endpoints and nodes with no match label",
			fields: fields{
				id: types.NamespacedName{
					Name: "test",
				},
				endpointSelectors: []*slimv1.LabelSelector{{
					MatchLabels: map[string]string{
						"app": "test",
					},
				}},
				nodeSelectors: []*slimv1.LabelSelector{{
					MatchLabels: map[string]string{
						"node-name": "node1",
					},
				}},
			},
			args: args{
				epDataStore: map[endpointID]*endpointMetadata{
					"123456": {
						id: "123456",
						labels: map[string]string{
							"app": "test",
						},
						nodeIP: "192.168.1.10",
					},
				},
				nodesAddresses2Labels: map[string]map[string]string{
					"192.168.1.10": {
						"bar": "bar",
					},
				},
			},
			want:           0,
			wantEndpointID: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &PolicyConfig{
				id:                tt.fields.id,
				endpointSelectors: getAsPolicyLabelSelectors(tt.fields.endpointSelectors),
				nodeSelectors:     getAsPolicyLabelSelectors(tt.fields.nodeSelectors),
				dstCIDRs:          tt.fields.dstCIDRs,
				excludedCIDRs:     tt.fields.excludedCIDRs,
				policyGwConfigs:   tt.fields.policyGwConfigs,
				matchedEndpoints:  tt.fields.matchedEndpoints,
				gatewayConfigs:    tt.fields.gatewayConfigs,
			}
			config.updateMatchedEndpointIDs(tt.args.epDataStore, tt.args.nodesAddresses2Labels)
			assert.Len(t, config.matchedEndpoints, tt.want)
			if tt.want > 0 {
				assert.Contains(t, config.matchedEndpoints, endpointID(tt.wantEndpointID))
			}
		})
	}
}

func TestParseCEGPTOS(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		wantTos     uint8
		wantTosSet  bool
	}{
		{
			name:       "no annotation",
			wantTos:    0,
			wantTosSet: false,
		},
		{
			name:        "decimal TOS",
			annotations: map[string]string{annotation.ServiceTOS: "184"},
			wantTos:     0xb8,
			wantTosSet:  true,
		},
		{
			name:        "hex TOS",
			annotations: map[string]string{annotation.ServiceTOS: "0xb8"},
			wantTos:     0xb8,
			wantTosSet:  true,
		},
		{
			name:        "zero TOS is pinned",
			annotations: map[string]string{annotation.ServiceTOS: "0"},
			wantTos:     0,
			wantTosSet:  true,
		},
		{
			name:        "invalid TOS is ignored",
			annotations: map[string]string{annotation.ServiceTOS: "banana"},
			wantTos:     0,
			wantTosSet:  false,
		},
		{
			name:        "TOS out of range is ignored",
			annotations: map[string]string{annotation.ServiceTOS: "256"},
			wantTos:     0,
			wantTosSet:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cegp := &v2.CiliumEgressGatewayPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "cegp-tos",
					Annotations: tt.annotations,
				},
				Spec: v2.CiliumEgressGatewayPolicySpec{
					Selectors: []v2.EgressRule{
						{
							PodSelector: &slimv1.LabelSelector{
								MatchLabels: map[string]string{"app": "test"},
							},
						},
					},
					DestinationCIDRs: []v2.CIDR{"0.0.0.0/0"},
					EgressGateway: &v2.EgressGateway{
						NodeSelector: &slimv1.LabelSelector{
							MatchLabels: map[string]string{"node": "gw"},
						},
					},
				},
			}

			config, err := ParseCEGP(hivetest.Logger(t), cegp)
			assert.NoError(t, err)
			assert.Equal(t, tt.wantTos, config.tos)
			assert.Equal(t, tt.wantTosSet, config.tosSet)
		})
	}
}
