// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package cmd

import (
	"fmt"
	"strconv"

	"github.com/cilium/cilium/api/v1/models"
	"github.com/cilium/cilium/pkg/annotation"
	"github.com/cilium/cilium/pkg/client"
)

type PodCniConfig struct {
	DeviceMtu int64
	RouteMtu  int64
	DeviceMac string
}

func getPodCniConfig(client *client.Client, daemonConf *models.DaemonConfigurationStatus, namespace, pod string) (*PodCniConfig, error) {
	conf := &PodCniConfig{}

	reply, err := client.GetPodAnnotations(namespace, pod)

	if err != nil {
		return nil, fmt.Errorf("unable to retrieve pod annotations from Cilium agent: %w", err)

	}

	if mtuStr, found := reply.Annotations[annotation.PodAnnotationMTU]; found {
		baseMtu, err := strconv.ParseInt(mtuStr, 10, 64)

		if err != nil {
			return nil, fmt.Errorf("error parsing MTU annotation: %w", err)
		}

		conf.DeviceMtu = baseMtu

		if baseMtu < daemonConf.RouteMTU {
			conf.RouteMtu = int64(baseMtu)
		}
	}

	if deviceMac, found := reply.Annotations[annotation.PodAnnotationMAC]; found && deviceMac != "" {
		conf.DeviceMac = deviceMac
	}

	return conf, nil
}
