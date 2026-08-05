// SPDX-License-Identifier: (GPL-2.0-only OR BSD-2-Clause)
/* Copyright Authors of Cilium */

#include <bpf/ctx/skb.h>
#include "common.h"
#include "pktgen.h"

#define ENABLE_IPV4

#define CLIENT_IP		v4_ext_one
#define CLIENT_PORT		__bpf_htons(8080)
#define FRONTEND_IP		v4_svc_two
#define FRONTEND_PORT		tcp_svc_one
#define BACKEND_IP		v4_pod_two
#define BACKEND_PORT		__bpf_htons(8080)

static volatile const __u8 *client_mac = mac_one;
static volatile const __u8 *lb_mac = mac_host;

#include "lib/bpf_lxc.h"

#include "lib/ipcache.h"
#include "lib/lb.h"

static __always_inline int csum_check(const struct __ctx_buff *ctx,
				      __u8 expected_tos)
{
	void *data, *data_end;
	__u32 *status_code;
	struct iphdr *l3;
	__u32 csum = 0;
	__u16 *w;
	int i;

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

	if (l3->tos != expected_tos)
		test_fatal("TOS mismatch, expected %#x, got %#x",
			   expected_tos, l3->tos);

	w = (__u16 *)l3;
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

	test_finish();
}

/* lb4_set_tos() must leave a valid IP checksum. */
PKTGEN("tc", "csum_tos")
int csum_tos_pktgen(struct __ctx_buff *ctx)
{
	struct pktgen builder;
	struct tcphdr *l4;
	void *data;

	pktgen__init(&builder, ctx);
	l4 = pktgen__push_ipv4_tcp_packet(
		&builder, (__u8 *)client_mac, (__u8 *)lb_mac, CLIENT_IP,
		FRONTEND_IP, CLIENT_PORT, FRONTEND_PORT);
	if (!l4)
		return TEST_ERROR;
	data = pktgen__push_data(&builder, default_data, sizeof(default_data));
	if (!data)
		return TEST_ERROR;
	pktgen__finish(&builder);

	return 0;
}

SETUP("tc", "csum_tos")
int csum_tos_setup(struct __ctx_buff *ctx)
{
	__u16 revnat_id = 1;

	lb_v4_add_service_tos(FRONTEND_IP, FRONTEND_PORT, IPPROTO_TCP, 1, revnat_id,
			      0xb8);
	lb_v4_add_backend(
		FRONTEND_IP, FRONTEND_PORT, 1, 124, BACKEND_IP, BACKEND_PORT,
		IPPROTO_TCP, 0);

	ipcache_v4_add_entry(BACKEND_IP, 0, 112233, 0, 0);

	pod_send_packet(ctx);

	return 0;
}

CHECK("tc", "csum_tos")
int csum_tos_check(const struct __ctx_buff *ctx)
{
	return csum_check(ctx, 0xb8);
}
