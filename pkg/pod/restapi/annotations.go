// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package restapi

import (
	"context"
	"log/slog"

	"github.com/cilium/cilium/api/v1/models"
	daemonapi "github.com/cilium/cilium/api/v1/server/restapi/daemon"
	k8sClient "github.com/cilium/cilium/pkg/k8s/client"
	"github.com/cilium/cilium/pkg/logging/logfields"
	"github.com/go-openapi/runtime/middleware"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type getAnnotationsHandler struct {
	logger *slog.Logger

	clientset k8sClient.Clientset
}

func (h *getAnnotationsHandler) Handle(params daemonapi.GetPodAnnotationsParams) middleware.Responder {
	h.logger.Debug(
		"GET /pod_annotations request",
		logfields.Params, params,
	)

	pod, err := h.clientset.CoreV1().Pods(params.Namespace).Get(context.Background(), params.Pod, v1.GetOptions{})

	if err != nil {
		if k8serrors.IsNotFound(err) {
			return daemonapi.NewGetPodAnnotationsNotFound()
		}

		return daemonapi.NewGetPodAnnotationsBadRequest().WithPayload(models.Error(err.Error()))
	}

	reply := &models.PodAnnotationsReply{
		Annotations: pod.Annotations,
	}

	return daemonapi.NewGetPodAnnotationsOK().WithPayload(reply)
}
