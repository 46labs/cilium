// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package cmd

import (
	"fmt"
	"net/netip"
	"os"
	"strconv"

	"github.com/spf13/cobra"

	"github.com/cilium/cilium/pkg/common"
	"github.com/cilium/cilium/pkg/maps/crap"
)

const (
	crapUpdateUsage = "Create/Update CRAP entry.\n"
)

var bpfCrapUpdateCmd = &cobra.Command{
	Args:    cobra.RangeArgs(2, 4),
	Use:     "update",
	Short:   "Update CRAP entries",
	Aliases: []string{"add"},
	Long:    crapUpdateUsage,
	Run: func(cmd *cobra.Command, args []string) {
		common.RequireRootPrivilege("cilium bpf crap update <public_ip> <pod_ip> [port_begin] [port_end]")

		m, err := crap.OpenPinnedCrapMap(log)
		if err != nil {
			Fatalf("Unable to open map: %s", err)
		}

		dst_ip, err := netip.ParseAddr(args[0])
		if err != nil {
			Fatalf("Unable to parse public IP '%s'", args[0])
		}
		if !dst_ip.Is4() {
			Fatalf("Only IPv4 public IPs are supported: '%s'", args[0])
		}

		pod_ip, err := netip.ParseAddr(args[1])
		if err != nil {
			Fatalf("Unable to parse pod IP '%s'", args[1])
		}
		if !pod_ip.Is4() {
			Fatalf("Only IPv4 pod IPs are supported: '%s'", args[1])
		}

		portBegin := uint16(0)
		portEnd := uint16(65535)

		if len(args) >= 3 {
			v, err := strconv.ParseUint(args[2], 10, 16)
			if err != nil {
				Fatalf("Unable to parse port_begin '%s'", args[2])
			}
			portBegin = uint16(v)
		}

		if len(args) >= 4 {
			v, err := strconv.ParseUint(args[3], 10, 16)
			if err != nil {
				Fatalf("Unable to parse port_end '%s'", args[3])
			}
			portEnd = uint16(v)
		}

		if err := m.UpdateCrapMapping(dst_ip, pod_ip, portBegin, portEnd); err != nil {
			fmt.Fprintf(os.Stderr, "error updating contents of map: %s\n", err)
			os.Exit(1)
		}
	},
}

func init() {
	BPFCrapCmd.AddCommand(bpfCrapUpdateCmd)
}
