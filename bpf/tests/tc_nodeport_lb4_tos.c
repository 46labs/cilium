// SPDX-License-Identifier: (GPL-2.0-only OR BSD-2-Clause)
/* Copyright Authors of Cilium */

#include <bpf/ctx/skb.h>
#include "common.h"
#include "pktgen.h"

/* Enable code paths under test */
#define ENABLE_IPV4
#define ENABLE_NODEPORT
#define ENABLE_HOST_ROUTING

/* Skip ingress policy checks */
#define USE_BPF_PROG_FOR_INGRESS_POLICY

#define CLIENT_IP	    v4_ext_one
#define CLIENT_PORT	    __bpf_htons(8080)

#define FRONTEND_IP	    v4_svc_two
#define FRONTEND_PORT	    __bpf_htons(8080)

#define BACKEND_IP	    v4_pod_two
#define BACKEND_PORT	    __bpf_htons(8080)

#define LB_IP		    v4_node_one
#define IPV4_DIRECT_ROUTING LB_IP

#define DEFAULT_IFACE	    24
#define BACKEND_IFACE	    25
#define BACKEND_EP_ID	    127

#define fib_lookup	    mock_fib_lookup

#define SERVICE_TOS	    0xb8
#define CLIENT_TOS	    0x20

static volatile const __u8 *client_mac = mac_one;
static volatile const __u8 *lb_mac = mac_host;
static volatile const __u8 *node_mac = mac_three;
static volatile const __u8 *local_backend_mac = mac_four;

long mock_fib_lookup(
	__maybe_unused void *ctx, struct bpf_fib_lookup *params,
	__maybe_unused int plen, __maybe_unused __u32 flags)
{
	params->ifindex = 0;

	__bpf_memcpy_builtin(params->smac, (__u8 *)lb_mac, ETH_ALEN);
	__bpf_memcpy_builtin(params->dmac, (__u8 *)client_mac, ETH_ALEN);

	return 0;
}

__section_entry
int mock_handle_policy(struct __ctx_buff *ctx __maybe_unused)
{
	return TC_ACT_REDIRECT;
}

struct {
	__uint(type, BPF_MAP_TYPE_PROG_ARRAY);
	__uint(key_size, sizeof(__u32));
	__uint(max_entries, 256);
	__array(values, int());
} mock_policy_call_map __section(".maps") = {
	.values = {
		[BACKEND_EP_ID] = &mock_handle_policy,
	},
};

#define tail_call_dynamic mock_tail_call_dynamic
static __always_inline __maybe_unused void
mock_tail_call_dynamic(struct __ctx_buff *ctx __maybe_unused,
		       const void *map __maybe_unused, __u32 slot __maybe_unused)
{
	tail_call(ctx, &mock_policy_call_map, slot);
}

#define ctx_redirect mock_ctx_redirect

static __always_inline __maybe_unused int
mock_ctx_redirect(const struct __sk_buff *ctx __maybe_unused,
		  int ifindex __maybe_unused, __u32 flags __maybe_unused)
{
	void *data = (void *)(long)ctx_data(ctx);
	void *data_end = (void *)(long)ctx->data_end;
	struct iphdr *ip4;

	ip4 = data + sizeof(struct ethhdr);
	if ((void *)ip4 + sizeof(*ip4) > data_end)
		return CTX_ACT_DROP;

	/* Forward to backend: */
	if (ip4->saddr == CLIENT_IP && ifindex == BACKEND_IFACE)
		return CTX_ACT_REDIRECT;

	return CTX_ACT_DROP;
}

#include "lib/bpf_host.h"

#include "lib/endpoint.h"
#include "lib/ipcache.h"
#include "lib/lb.h"

ASSIGN_CONFIG(__u32, interface_ifindex, DEFAULT_IFACE)

static __always_inline void lb_tos_setup(struct __ctx_buff *ctx, __u8 tos,
					 __u8 proto)
{
	__u16 revnat_id = 1;

	lb_v4_add_service_tos(FRONTEND_IP, FRONTEND_PORT, proto, 1, revnat_id,
			      tos);
	lb_v4_add_backend(
		FRONTEND_IP, FRONTEND_PORT, 1, 124, BACKEND_IP, BACKEND_PORT,
		proto, 0);

	/* add local backend */
	endpoint_v4_add_entry(BACKEND_IP, BACKEND_IFACE, BACKEND_EP_ID, 0, 0, 0,
			      (__u8 *)local_backend_mac, (__u8 *)node_mac);

	ipcache_v4_add_entry(BACKEND_IP, 0, 112233, 0, 0);

	netdev_receive_packet(ctx);
}

static __always_inline int lb_tos_check(__maybe_unused const struct __ctx_buff *ctx,
					__u8 expected_tos, __u8 proto)
{
	void *data, *data_end;
	__u32 *status_code;
	struct tcphdr *l4_tcp;
	struct udphdr *l4_udp;
	struct ethhdr *l2;
	struct iphdr *l3;
	__u32 l4_size = proto == IPPROTO_UDP ? sizeof(struct udphdr) :
						sizeof(struct tcphdr);

	test_init();

	data = (void *)(long)ctx_data(ctx);
	data_end = (void *)(long)ctx->data_end;

	if (data + sizeof(__u32) > data_end)
		test_fatal("status code out of bounds");

	status_code = data;

	assert(*status_code == CTX_ACT_REDIRECT);

	l2 = data + sizeof(__u32);
	if ((void *)l2 + sizeof(struct ethhdr) > data_end)
		test_fatal("l2 out of bounds");

	l3 = (void *)l2 + sizeof(struct ethhdr);
	if ((void *)l3 + sizeof(struct iphdr) > data_end)
		test_fatal("l3 out of bounds");

	l4_tcp = (void *)l3 + sizeof(*l3);
	if ((void *)l4_tcp + l4_size > data_end)
		test_fatal("l4 out of bounds");

	if (memcmp(l2->h_source, (__u8 *)node_mac, ETH_ALEN) != 0)
		test_fatal("src MAC is not the node MAC")
	if (memcmp(l2->h_dest, (__u8 *)local_backend_mac, ETH_ALEN) != 0)
		test_fatal("dst MAC is not the endpoint MAC")

	if (l3->saddr != CLIENT_IP)
		test_fatal("src IP has changed");

	if (l3->daddr != BACKEND_IP)
		test_fatal("dst IP hasn't been NATed to local backend IP");

	if (l3->tos != expected_tos)
		test_fatal("TOS mismatch, expected %#x, got %#x",
			   expected_tos, l3->tos);

	if (proto == IPPROTO_UDP) {
		l4_udp = (void *)l3 + sizeof(*l3);
		if (l4_udp->source != CLIENT_PORT)
			test_fatal("src port has changed");

		if (l4_udp->dest != BACKEND_PORT)
			test_fatal("dst UDP port hasn't been NATed to backend port");
	} else {
		if (l4_tcp->source != CLIENT_PORT)
			test_fatal("src port has changed");

		if (l4_tcp->dest != BACKEND_PORT)
			test_fatal("dst TCP port hasn't been NATed to backend port");
	}

	test_finish();
}

static __always_inline int lb_tos_pktgen(struct __ctx_buff *ctx, __u8 proto,
					 __u8 tos)
{
	struct pktgen builder;
	struct iphdr *l3;
	struct tcphdr *l4_tcp;
	struct udphdr *l4_udp;
	void *data;

	pktgen__init(&builder, ctx);

	if (proto == IPPROTO_UDP) {
		l4_udp = pktgen__push_ipv4_udp_packet(
			&builder, (__u8 *)client_mac, (__u8 *)lb_mac, CLIENT_IP,
			FRONTEND_IP, CLIENT_PORT, FRONTEND_PORT);
		if (!l4_udp)
			return TEST_ERROR;
		l3 = (void *)l4_udp - sizeof(struct iphdr);
	} else {
		l4_tcp = pktgen__push_ipv4_tcp_packet(
			&builder, (__u8 *)client_mac, (__u8 *)lb_mac, CLIENT_IP,
			FRONTEND_IP, CLIENT_PORT, FRONTEND_PORT);
		if (!l4_tcp)
			return TEST_ERROR;
		l3 = (void *)l4_tcp - sizeof(struct iphdr);
	}

	l3->tos = tos;

	data = pktgen__push_data(&builder, "hello", 5);
	if (!data)
		return TEST_ERROR;

	pktgen__finish(&builder);

	return 0;
}

/* Test that a request to a service without a pinned TOS keeps the TOS set by
 * the client.
 */
PKTGEN("tc", "lb4_tos_preserve")
int lb4_tos_preserve_pktgen(struct __ctx_buff *ctx)
{
	return lb_tos_pktgen(ctx, IPPROTO_TCP, CLIENT_TOS);
}

SETUP("tc", "lb4_tos_preserve")
int lb4_tos_preserve_setup(struct __ctx_buff *ctx)
{
	lb_tos_setup(ctx, 0, IPPROTO_TCP);

	return 0;
}

CHECK("tc", "lb4_tos_preserve")
int lb4_tos_preserve_check(__maybe_unused const struct __ctx_buff *ctx)
{
	return lb_tos_check(ctx, CLIENT_TOS, IPPROTO_TCP);
}

/* Test that the TOS pinned by the service overrides the client's TOS. */
PKTGEN("tc", "lb4_tos_override")
int lb4_tos_override_pktgen(struct __ctx_buff *ctx)
{
	return lb_tos_pktgen(ctx, IPPROTO_TCP, CLIENT_TOS);
}

SETUP("tc", "lb4_tos_override")
int lb4_tos_override_setup(struct __ctx_buff *ctx)
{
	lb_tos_setup(ctx, SERVICE_TOS, IPPROTO_TCP);

	return 0;
}

CHECK("tc", "lb4_tos_override")
int lb4_tos_override_check(__maybe_unused const struct __ctx_buff *ctx)
{
	return lb_tos_check(ctx, SERVICE_TOS, IPPROTO_TCP);
}

/* ---- UDP ---- */

/* Test that a UDP request to a service without a pinned TOS keeps the TOS set
 * by the client.
 */
PKTGEN("tc", "lb4_tos_udp_preserve")
int lb4_tos_udp_preserve_pktgen(struct __ctx_buff *ctx)
{
	return lb_tos_pktgen(ctx, IPPROTO_UDP, CLIENT_TOS);
}

SETUP("tc", "lb4_tos_udp_preserve")
int lb4_tos_udp_preserve_setup(struct __ctx_buff *ctx)
{
	lb_tos_setup(ctx, 0, IPPROTO_UDP);

	return 0;
}

CHECK("tc", "lb4_tos_udp_preserve")
int lb4_tos_udp_preserve_check(__maybe_unused const struct __ctx_buff *ctx)
{
	return lb_tos_check(ctx, CLIENT_TOS, IPPROTO_UDP);
}

/* Test that the TOS pinned by the service overrides the client's TOS for UDP. */
PKTGEN("tc", "lb4_tos_udp_override")
int lb4_tos_udp_override_pktgen(struct __ctx_buff *ctx)
{
	return lb_tos_pktgen(ctx, IPPROTO_UDP, CLIENT_TOS);
}

SETUP("tc", "lb4_tos_udp_override")
int lb4_tos_udp_override_setup(struct __ctx_buff *ctx)
{
	lb_tos_setup(ctx, SERVICE_TOS, IPPROTO_UDP);

	return 0;
}

CHECK("tc", "lb4_tos_udp_override")
int lb4_tos_udp_override_check(__maybe_unused const struct __ctx_buff *ctx)
{
	return lb_tos_check(ctx, SERVICE_TOS, IPPROTO_UDP);
}
