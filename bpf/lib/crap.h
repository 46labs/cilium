/* SPDX-License-Identifier: (GPL-2.0-only OR BSD-2-Clause) */
/* Copyright Authors of Cilium */

#pragma once

#include <linux/bpf.h>
#include <bpf/section.h>
#include <bpf/loader.h>
#include <lib/static_data.h>

struct crap_key {
	__u32 dst_ip;
};

struct crap_value {
	__u32 pod_ip;
	__u16 port_begin;
	__u16 port_end;
};

static __always_inline bool
crap_port_match(__u16 dport, const struct crap_value *val)
{
	return dport >= val->port_begin && dport <= val->port_end;
}

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, struct crap_key);
	__type(value, struct crap_value);
	__uint(pinning, LIBBPF_PIN_BY_NAME);
	__uint(max_entries, 8192);
	__uint(map_flags, BPF_F_NO_PREALLOC);
} cilium_crap_map __section_maps_btf;
