// SPDX-License-Identifier: (GPL-2.0-only OR BSD-2-Clause)
/* Only meaningful when the SIP datapath extensions are compiled out. */
#ifndef ENABLE_SIP_INSPECTION

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
#include "lib/egressgw_policy.h"
#include "lib/hash.h"

CHECK("tc", "sip_disabled_state")
int sip_disabled_state(__maybe_unused struct __ctx_buff *ctx)
{
	test_init();

	TEST("map_abi_preserved", {
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
		/* The Call-ID field is not restored when the switch is off. */
		assert(hash == 0x87654321);
	});

	TEST("sip_flags_not_propagated_from_egress_gateway_policy", {
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
		__u8 sip_needed = 0;
		__u16 sip_port = 0;

		assert(map_update_elem(&cilium_egress_gw_policy_v4, &key, &policy, BPF_ANY) == 0);
		assert(egress_gw_snat_needed(v4_pod_one, v4_ext_one, 0, &snat_addr,
					    &ifindex, &sip_needed, &sip_port));
		assert(snat_addr == policy.egress_ip);
		assert(sip_needed == 0);
		assert(sip_port == 0);
	});

	test_finish();
}

/* A reply to an outside-initiated connection must still bypass the egress
 * gateway when the SIP datapath extensions are disabled. The reply check
 * used to be compiled out together with the SIP exception, which sent such
 * replies through egress_gw_handle_packet().
 */
static __always_inline int add_egw_reply_ct(struct __ctx_buff *ctx)
{
	struct ipv4_ct_tuple tuple = {};
	struct iphdr *ip4;
	void *data, *data_end;
	fraginfo_t fraginfo;
	int l4_off;

	if (!revalidate_data(ctx, &data, &data_end, &ip4))
		return TEST_ERROR;
	fraginfo = ipfrag_encode_ipv4(ip4);
	snat_v4_init_tuple(ip4, NAT_DIR_EGRESS, &tuple);
	l4_off = ETH_HLEN + ipv4_hdrlen(ip4);
	if (ct_extract_ports4(ctx, ip4, fraginfo, l4_off, CT_EGRESS, &tuple))
		return TEST_ERROR;
	tuple.flags = TUPLE_F_IN;
	return ct_create4(get_ct_map4(&tuple), NULL, &tuple, ctx,
			  CT_INGRESS, NULL, NULL);
}

PKTGEN("tc", "sip_disabled_egw_reply")
int sip_disabled_egw_reply_pktgen(struct __ctx_buff *ctx)
{
	struct pktgen builder;
	struct udphdr *udp;
	void *data;

	pktgen__init(&builder, ctx);
	udp = pktgen__push_ipv4_udp_packet(&builder,
					   (__u8 *)mac_one, (__u8 *)mac_two,
					   v4_pod_one, v4_ext_one,
					   __bpf_htons(5060), __bpf_htons(5070));
	if (!udp)
		return TEST_ERROR;

	data = pktgen__push_data(&builder, "x", 1);
	if (!data)
		return TEST_ERROR;

	pktgen__finish(&builder);
	return 0;
}

CHECK("tc", "sip_disabled_egw_reply")
int sip_disabled_egw_reply_check(struct __ctx_buff *ctx)
{
	const struct egress_gw_policy_entry *policy;
	struct trace_ctx trace = {
		.reason = TRACE_REASON_UNKNOWN,
		.monitor = 0,
	};
	int ret;

	test_init();

	add_egressgw_policy_entry(v4_pod_one, v4_ext_one, 32,
				  v4_node_two, v4_node_one);
	policy = lookup_ip4_egress_gw_policy(v4_pod_one, v4_ext_one, 0);
	assert(policy != NULL);
	assert(add_egw_reply_ct(ctx) == 0);

	ret = egress_gw_handle_request(ctx, bpf_htons(ETH_P_IP), 123456,
				       WORLD_ID, &trace);
	assert(ret == CTX_ACT_OK);

	test_finish();
}

#endif /* !ENABLE_SIP_INSPECTION */
