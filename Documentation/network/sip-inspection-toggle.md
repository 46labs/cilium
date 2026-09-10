# Disabling the 46labs SIP datapath extensions

This fork exposes an agent startup option, `enable-sip-inspection` (default
`true`). The agent emits a compile-time `ENABLE_SIP_INSPECTION` define and
compiles the SIP datapath extensions only when the option is enabled. With a
Cilium image containing this change, set the Helm value:

```yaml
enableSIPInspection: false
```

For agents not managed by this chart, use `--enable-sip-inspection=false`,
`CILIUM_ENABLE_SIP_INSPECTION=false`, or the ConfigMap entry
`enable-sip-inspection: "false"`. This is a node-wide compile-time setting for
the BPF programs, not a per-Service setting or a live switch. It takes effect
as agents restart and recompile/regenerate their datapath. The Cilium container
image is unchanged; each node's agent compiles the appropriate programs from
the same image.

When disabled, on **all** paths (including internal pod-to-pod traffic):

- The SIP parser and its payload-scanning implementation are omitted from the
  generated BPF programs. The remaining no-op helper returns zero without
  reading or linearizing payloads. Long `Call-ID:` and compact `i:` headers are
  treated identically.
- IPv4 CT/NAT keys have a zero Call-ID field. Distinct calls on the same UDP
  tuple share ordinary UDP state; backend hashing ignores Call-ID.
- The SIP timeout override (`bpf-ct-timeout-sip`, normally 1200 seconds) is
  ignored. Configured ordinary UDP and UDP-Service timeouts apply instead.
- Existing SIP flags in Service and egress-gateway BPF maps are ignored by the
  datapath. No SIP-specific reverse-NAT dispatch, remote-backend shortcut, or
  fixed SIP source-port allocation is used.
- Fragment tracking still restores UDP ports, but cannot restore an old SIP
  hash from the pinned fragment map.

Kubernetes annotations and their stored map flags are intentionally retained
but inert while this switch is off; they are not deleted or rewritten. Normal
Cilium routing, policy enforcement, Service LB, egress-gateway NAT, TOS and
explicit service pinning remain available. This does **not** remove Cilium as
the CNI and does not change direct BGP/Multus interfaces. If a node must later
run a workload that needs the SIP extensions, enable the option for that node
and regenerate all affected datapath programs; enabled and disabled programs
are not supported concurrently under one Cilium agent.

## Transition precautions

Preserve the existing map layouts (including the unused Call-ID field). Do not
resize, flush, or recreate maps as part of this patch. Preserving the ABI does
**not** make active SIP state compatible across modes: old nonzero-hash CT/NAT
entries no longer match the new zero-hash tuples. They remain until normal GC
or LRU eviction; disabling the switch does not instantly empty the table.

Drain affected calls before switching. Coordinate the change across every
Cilium node carrying these flows; avoid sending live calls through a mixture
of enabled and disabled datapaths. The same precaution applies to rollback.
Reload/regenerate existing endpoint programs too, and verify the effective
agent setting and BPF configuration, not just the Helm values. No rollout or
map cleanup is performed by the code in this patch.

## Verification

Run the existing `sip`, `sip_lb`, and `sip_egw_reverse` BPF suites with the
default enabled setting. `sip_disabled_parser` reuses the SIP payload corpus
with inspection disabled; `sip_disabled` checks timeout handling, backend
hashing, map ABI, old fragment entries, and old egress-gateway SIP flags.
Before production rollout also verify ordinary UDP/DNS request/reply traffic,
Service NAT, and fragmented SIP on a staging cluster with the new agent image.
