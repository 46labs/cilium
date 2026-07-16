/* SPDX-License-Identifier: (GPL-2.0-only OR BSD-2-Clause) */
/* Copyright Authors of Cilium */

#pragma once

#include <linux/bpf.h>
#include <bpf/section.h>
#include <bpf/loader.h>
#include <lib/static_data.h>

#define MAX_CRAP_RULES_PER_IP 8

struct crap_key {
	__u32 dst_ip;
};

struct crap_rule {
	__u32 pod_ip;
	__u16 port_begin;
	__u16 port_end;
};

struct crap_value {
	struct crap_rule rules[MAX_CRAP_RULES_PER_IP];
};

static __always_inline bool
crap_rule_is_valid(const struct crap_rule *rule)
{
	return rule->pod_ip != 0;
}

static __always_inline bool
crap_rule_port_match(__u16 dport, const struct crap_rule *rule)
{
	return dport >= rule->port_begin && dport <= rule->port_end;
}

static __always_inline bool
crap_check_proto_and_load_dport(struct __ctx_buff *ctx, const struct iphdr *ip4,
				__u16 *dport)
{
	__be16 dport_be;

	if (ip4->protocol != IPPROTO_TCP && ip4->protocol != IPPROTO_UDP)
		return false;

	if (l4_load_port(ctx, ETH_HLEN + ipv4_hdrlen(ip4) + TCP_DPORT_OFF, &dport_be))
		return false;

	*dport = bpf_ntohs(dport_be);
	return true;
}

static __always_inline bool
crap_value_has_any_rule(const struct crap_value *val)
{
#pragma unroll
	for (int i = 0; i < MAX_CRAP_RULES_PER_IP; i++) {
		if (crap_rule_is_valid(&val->rules[i]))
			return true;
	}
	return false;
}

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, struct crap_key);
	__type(value, struct crap_value);
	__uint(pinning, LIBBPF_PIN_BY_NAME);
	__uint(max_entries, 8192);
	__uint(map_flags, BPF_F_NO_PREALLOC);
} cilium_crap_map __section_maps_btf;
