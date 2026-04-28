// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package builder

import (
	"github.com/cilium/cilium/cilium-cli/connectivity/check"
	"github.com/cilium/cilium/cilium-cli/connectivity/tests"
	"github.com/cilium/cilium/cilium-cli/utils/features"
)

type egressGatewaySipPinning struct{}

func (t egressGatewaySipPinning) build(ct *check.ConnectivityTest, _ map[string]string) {
	newTest("seq-sip-lb-maglev-egw-pinning", ct).
		WithCondition(func() bool { return ct.Params().IncludeUnsafeTests }).
		WithIPRoutesFromOutsideToPodCIDRs().
		WithFeatureRequirements(
			features.RequireEnabled(features.EgressGateway),
			features.RequireEnabled(features.NodeWithoutCilium),
		).
		WithScenarios(tests.SipEgressGateway())
}
