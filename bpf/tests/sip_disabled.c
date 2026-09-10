// SPDX-License-Identifier: (GPL-2.0-only OR BSD-2-Clause)
/* Copyright 46labs LLC */

#include <bpf/ctx/skb.h>

#define ktime_get_ns() (1000ULL * NSEC_PER_SEC)
#define jiffies64() 1000ULL

#include "common.h"
#include "pktgen.h"

#define ENABLE_IPV4 1
#define ENABLE_IPV4_FRAGMENTS 1
#define ENABLE_NODEPORT 1
#define ENABLE_EGRESS_GATEWAY 1
#define ENABLE_MASQUERADE_IPV4 1
#define ENCAP_IFINDEX 42

#include "lib/bpf_host.h"
#include "lib/hash.h"

CHECK("tc", "sip_disabled_state")
int sip_disabled_state(__maybe_unused struct __ctx_buff *ctx)
{
	test_init();

	TEST("switch_off", {
		assert(!sip_inspection_enabled());
		/* Keep pinned map ABI, even though the Call-ID field is unused. */
		assert(sizeof(struct ipv4_ct_tuple) == 20);
		assert(sizeof(struct ct_entry) == 56);
	});

	TEST("old_sip_entry_uses_normal_udp_timeout", {
		struct ct_entry entry = { .is_sip = 1 };
		union tcp_flags flags = {};

		ct_update_timeout(&entry, false, CT_EGRESS, flags);
		assert(entry.lifetime == 1000 + CT_CONNECTION_LIFETIME_NONTCP);
		ct_update_timeout(&entry, false, CT_SERVICE, flags);
		assert(entry.lifetime == 1000 + CT_SERVICE_LIFETIME_NONTCP);
	});

	TEST("call_id_does_not_change_backend_hash", {
		struct ipv4_ct_tuple tuple = ((struct ipv4_ct_tuple){
			.saddr = v4_pod_one,
			.daddr = v4_ext_one,
			.sport = __bpf_htons(5060),
			.dport = __bpf_htons(5070),
			.nexthdr = IPPROTO_UDP,
		});
		__u32 expected = hash_from_tuple_v4(&tuple);

		tuple.sip_call_id_hash = 0x12345678;
		assert(hash_from_tuple_v4(&tuple) == expected);
	});

	TEST("old_fragment_restores_ports_but_not_call_id", {
		struct ipv4_frag_id key = ((struct ipv4_frag_id){
			.saddr = v4_pod_one,
			.daddr = v4_ext_one,
			.id = __bpf_htons(7),
			.proto = IPPROTO_UDP,
		});
		struct ipv4_frag_l4ports old = ((struct ipv4_frag_l4ports){
			.sport = __bpf_htons(5060),
			.dport = __bpf_htons(5070),
			.sip_call_id_hash = 0x12345678,
		});
		struct ipv4_frag_l4ports ports = {};
		__u32 hash = 0x87654321;

		assert(map_update_elem(&cilium_ipv4_frag_datagrams, &key, &old, BPF_ANY) == 0);
		assert(ipv4_frag_get_l4ports(&key, &ports, &hash) == 0);
		assert(ports.sport == old.sport);
		assert(ports.dport == old.dport);
		assert(hash == 0);
	});

	TEST("old_sip_policy_keeps_normal_egress_gateway", {
		struct egress_gw_policy_key key = ((struct egress_gw_policy_key){
			.lpm_key = { EGRESS_PREFIX_LEN_V4(32), {} },
			.saddr = v4_pod_one,
			.daddr = v4_ext_one,
		});
		struct egress_gw_policy_entry policy = ((struct egress_gw_policy_entry){
			.egress_ip = v4_node_one,
			.gateway_ip = v4_node_two,
			.sip_port = 5060,
			.sip_inspect = 1,
		});
		__be32 snat_addr = 0;
		__u32 ifindex = 0;
		__u8 sip_needed = 1;
		__u16 sip_port = 0;

		assert(map_update_elem(&cilium_egress_gw_policy_v4, &key, &policy, BPF_ANY) == 0);
		assert(!egress_gw_sip_inspection_needed(v4_pod_one, v4_ext_one, 0, &sip_port));
		assert(egress_gw_snat_needed(v4_pod_one, v4_ext_one, 0, &snat_addr,
					    &ifindex, &sip_needed, &sip_port));
		assert(snat_addr == policy.egress_ip);
		assert(sip_needed == 0);
		assert(sip_port == 0);
	});

	test_finish();
}
