// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package client

import (
	"github.com/cilium/cilium/api/v1/client/daemon"
	"github.com/cilium/cilium/api/v1/models"
)

// GetPodAnnotations returns pod annotations.
func (c *Client) GetPodAnnotations(namespace, pod string) (*models.PodAnnotationsReply, error) {
	resp, err := c.Daemon.GetPodAnnotations(daemon.NewGetPodAnnotationsParams().WithNamespace(namespace).WithPod(pod))
	if err != nil {
		return nil, Hint(err)
	}
	return resp.Payload, nil
}
