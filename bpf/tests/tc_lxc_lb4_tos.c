// SPDX-License-Identifier: (GPL-2.0-only OR BSD-2-Clause)
/* Copyright Authors of Cilium */

#include <bpf/ctx/skb.h>
#include "common.h"
#include "pktgen.h"

/* Enable code paths under test */
#define ENABLE_IPV4

#define CLIENT_IP		v4_ext_one
#define CLIENT_PORT		__bpf_htons(8080)

#define FRONTEND_IP		v4_svc_two
#define FRONTEND_PORT		tcp_svc_one

#define BACKEND_IP		v4_pod_two
#define BACKEND_PORT		__bpf_htons(8080)

#define SERVICE_TOS		0xb8
#define CLIENT_TOS		0x20

#define FRONTEND_PORT_UDP	__bpf_htons(8080)

static volatile const __u8 *client_mac = mac_one;
static volatile const __u8 *lb_mac = mac_host;

#include "lib/bpf_lxc.h"

#include "lib/ipcache.h"
#include "lib/lb.h"

static __always_inline void lxc_lb4_tos_setup(struct __ctx_buff *ctx, __u8 tos,
					      __u8 proto, __u16 frontend_port)
{
	__u16 revnat_id = 1;

	lb_v4_add_service_tos(FRONTEND_IP, frontend_port, proto, 1, revnat_id,
			      tos);
	lb_v4_add_backend(
		FRONTEND_IP, frontend_port, 1, 124, BACKEND_IP, BACKEND_PORT,
		proto, 0);

	ipcache_v4_add_entry(BACKEND_IP, 0, 112233, 0, 0);

	pod_send_packet(ctx);
}

static __always_inline int lxc_lb4_tos_check(__maybe_unused const struct __ctx_buff *ctx,
					     __u8 expected_tos, __u8 proto)
{
	void *data, *data_end;
	__u32 *status_code;
	struct tcphdr *l4_tcp;
	struct udphdr *l4_udp;
	struct iphdr *l3;
	__u32 l4_size = proto == IPPROTO_UDP ? sizeof(struct udphdr) :
						sizeof(struct tcphdr);

	test_init();

	data = (void *)(long)ctx_data(ctx);
	data_end = (void *)(long)ctx->data_end;

	if (data + sizeof(__u32) > data_end)
		test_fatal("status code out of bounds");

	status_code = data;

	test_log("Status code: %d", *status_code);

	l3 = data + sizeof(__u32) + sizeof(struct ethhdr);
	if ((void *)l3 + sizeof(struct iphdr) > data_end)
		test_fatal("l3 out of bounds");

	l4_tcp = (void *)l3 + sizeof(*l3);
	if ((void *)l4_tcp + l4_size > data_end)
		test_fatal("l4 out of bounds");

	if (l3->saddr != CLIENT_IP)
		test_fatal("src IP has changed");

	if (l3->daddr != BACKEND_IP)
		test_fatal("dst IP hasn't been NATed to remote backend IP");

	if (l3->tos != expected_tos)
		test_fatal("TOS mismatch, expected %#x, got %#x",
			   expected_tos, l3->tos);

	/* The TOS rewrite must not corrupt the IP checksum. */
	{
		__u32 csum = 0;
		__u16 *w = (__u16 *)l3;
		int i;

		for (i = 0; i < 10; i++) {
			if (i == 5)
				continue;
			csum += bpf_ntohs(w[i]);
		}
		while (csum >> 16)
			csum = (csum & 0xffff) + (csum >> 16);
		csum = ~csum & 0xffff;
		if (csum != bpf_ntohs(l3->check))
			test_fatal("IP header checksum mismatch: stored=%u computed=%u",
				   (__u32)bpf_ntohs(l3->check), (__u32)csum);
	}

	if (proto == IPPROTO_UDP) {
		l4_udp = (void *)l3 + sizeof(*l3);
		if (l4_udp->source != CLIENT_PORT)
			test_fatal("src port has changed");

		if (l4_udp->dest != BACKEND_PORT)
			test_fatal("dst port hasn't been NATed to backend port");
	} else {
		if (l4_tcp->source != CLIENT_PORT)
			test_fatal("src port has changed");

		if (l4_tcp->dest != BACKEND_PORT)
			test_fatal("dst port hasn't been NATed to backend port");
	}

	test_finish();
}

static __always_inline int lxc_lb4_tos_pktgen(struct __ctx_buff *ctx,
					       __u8 proto, __u8 tos,
					       __be16 frontend_port)
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
			FRONTEND_IP, CLIENT_PORT, frontend_port);
		if (!l4_udp)
			return TEST_ERROR;
		l3 = (void *)l4_udp - sizeof(struct iphdr);
	} else {
		l4_tcp = pktgen__push_ipv4_tcp_packet(
			&builder, (__u8 *)client_mac, (__u8 *)lb_mac, CLIENT_IP,
			FRONTEND_IP, CLIENT_PORT, frontend_port);
		if (!l4_tcp)
			return TEST_ERROR;
		l3 = (void *)l4_tcp - sizeof(struct iphdr);
	}

	l3->tos = tos;

	data = pktgen__push_data(&builder, default_data, sizeof(default_data));
	if (!data)
		return TEST_ERROR;

	pktgen__finish(&builder);

	return 0;
}

/* Test that a request to a cluster IP service without a pinned TOS keeps the
 * TOS set by the client.
 */
PKTGEN("tc", "lxc_lb4_tos_preserve")
int lxc_lb4_tos_preserve_pktgen(struct __ctx_buff *ctx)
{
	return lxc_lb4_tos_pktgen(ctx, IPPROTO_TCP, CLIENT_TOS, FRONTEND_PORT);
}

SETUP("tc", "lxc_lb4_tos_preserve")
int lxc_lb4_tos_preserve_setup(struct __ctx_buff *ctx)
{
	lxc_lb4_tos_setup(ctx, 0, IPPROTO_TCP, FRONTEND_PORT);

	return 0;
}

CHECK("tc", "lxc_lb4_tos_preserve")
int lxc_lb4_tos_preserve_check(__maybe_unused const struct __ctx_buff *ctx)
{
	return lxc_lb4_tos_check(ctx, CLIENT_TOS, IPPROTO_TCP);
}

/* Test that the TOS pinned by the service overrides the client's TOS. */
PKTGEN("tc", "lxc_lb4_tos_override")
int lxc_lb4_tos_override_pktgen(struct __ctx_buff *ctx)
{
	return lxc_lb4_tos_pktgen(ctx, IPPROTO_TCP, CLIENT_TOS, FRONTEND_PORT);
}

SETUP("tc", "lxc_lb4_tos_override")
int lxc_lb4_tos_override_setup(struct __ctx_buff *ctx)
{
	lxc_lb4_tos_setup(ctx, SERVICE_TOS, IPPROTO_TCP, FRONTEND_PORT);

	return 0;
}

CHECK("tc", "lxc_lb4_tos_override")
int lxc_lb4_tos_override_check(__maybe_unused const struct __ctx_buff *ctx)
{
	return lxc_lb4_tos_check(ctx, SERVICE_TOS, IPPROTO_TCP);
}

/* ---- UDP ---- */

/* Test that a UDP request to a cluster IP service without a pinned TOS keeps
 * the TOS set by the client.
 */
PKTGEN("tc", "lxc_lb4_tos_udp_preserve")
int lxc_lb4_tos_udp_preserve_pktgen(struct __ctx_buff *ctx)
{
	return lxc_lb4_tos_pktgen(ctx, IPPROTO_UDP, CLIENT_TOS, FRONTEND_PORT_UDP);
}

SETUP("tc", "lxc_lb4_tos_udp_preserve")
int lxc_lb4_tos_udp_preserve_setup(struct __ctx_buff *ctx)
{
	lxc_lb4_tos_setup(ctx, 0, IPPROTO_UDP, FRONTEND_PORT_UDP);

	return 0;
}

CHECK("tc", "lxc_lb4_tos_udp_preserve")
int lxc_lb4_tos_udp_preserve_check(__maybe_unused const struct __ctx_buff *ctx)
{
	return lxc_lb4_tos_check(ctx, CLIENT_TOS, IPPROTO_UDP);
}

/* Test that the TOS pinned by the service overrides the client's TOS for UDP. */
PKTGEN("tc", "lxc_lb4_tos_udp_override")
int lxc_lb4_tos_udp_override_pktgen(struct __ctx_buff *ctx)
{
	return lxc_lb4_tos_pktgen(ctx, IPPROTO_UDP, CLIENT_TOS, FRONTEND_PORT_UDP);
}

SETUP("tc", "lxc_lb4_tos_udp_override")
int lxc_lb4_tos_udp_override_setup(struct __ctx_buff *ctx)
{
	lxc_lb4_tos_setup(ctx, SERVICE_TOS, IPPROTO_UDP, FRONTEND_PORT_UDP);

	return 0;
}

CHECK("tc", "lxc_lb4_tos_udp_override")
int lxc_lb4_tos_udp_override_check(__maybe_unused const struct __ctx_buff *ctx)
{
	return lxc_lb4_tos_check(ctx, SERVICE_TOS, IPPROTO_UDP);
}
