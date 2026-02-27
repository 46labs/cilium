// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package restapi

import (
	"log/slog"

	daemonapi "github.com/cilium/cilium/api/v1/server/restapi/daemon"
	k8sClient "github.com/cilium/cilium/pkg/k8s/client"
	"github.com/cilium/hive/cell"
)

var Cell = cell.Module(
	"pod-info",
	"Pod info requests via REST API",

	cell.Provide(newPodAnnotationsApiHandler),
)

type podInfoApiHandlerParams struct {
	cell.In

	Logger    *slog.Logger
	Clientset k8sClient.Clientset
}

type podInfoApiHandlerOut struct {
	cell.Out

	GetAnnotationsHandler daemonapi.GetPodAnnotationsHandler
}

func newPodAnnotationsApiHandler(params podInfoApiHandlerParams) podInfoApiHandlerOut {
	return podInfoApiHandlerOut{
		GetAnnotationsHandler: &getAnnotationsHandler{
			logger:    params.Logger,
			clientset: params.Clientset,
		},
	}
}
