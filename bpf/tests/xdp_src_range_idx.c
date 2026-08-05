// SPDX-License-Identifier: (GPL-2.0-only OR BSD-2-Clause)
/* Copyright Authors of Cilium */

#include "bpf/ctx/xdp.h"
#include "common.h"
#include "pktgen.h"

/* Enable code paths under test */
#define ENABLE_IPV4
#define ENABLE_NODEPORT
#define ENABLE_NODEPORT_ACCELERATION

/* Enable the per-service algorithm dispatch that includes the source range
 * index backend selection.
 */
#define LB_SELECTION_PER_SERVICE

/* Skip ingress policy checks, not needed to validate hairpin flow */
#define USE_BPF_PROG_FOR_INGRESS_POLICY

#define fib_lookup mock_fib_lookup

static const char fib_smac[6] = {0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02};
static const char fib_dmac[6] = {0x13, 0x37, 0x13, 0x37, 0x13, 0x37};

long mock_fib_lookup(__maybe_unused void *ctx, struct bpf_fib_lookup *params,
		     __maybe_unused int plen, __maybe_unused __u32 flags)
{
	__bpf_memcpy_builtin(params->smac, fib_smac, ETH_ALEN);
	__bpf_memcpy_builtin(params->dmac, fib_dmac, ETH_ALEN);
	return 0;
}

#include "lib/bpf_xdp.h"
#include "lib/lb.h"

#define CLIENT_IP	IPV4(10, 0, 0, 1)
#define CLIENT_CIDR	IPV4(10, 0, 0, 0)
#define CLIENT_PORT	__bpf_htons(23445)
#define CLIENT_PORT2	__bpf_htons(23446)
#define CLIENT_PORT3	__bpf_htons(23447)

#define FRONTEND_IP	IPV4(10, 0, 1, 1)
#define FRONTEND_PORT	__bpf_htons(80)

#define BACKEND_IP1	IPV4(10, 0, 2, 1)
#define BACKEND_IP2	IPV4(10, 0, 3, 1)
#define BACKEND_PORT	__bpf_htons(8080)

#define REV_NAT_INDEX	123
#define BACKEND_ID1	7
#define BACKEND_ID2	42

static volatile const __u8 *client_mac = mac_one;
static volatile const __u8 *lb_mac = mac_two;

static __always_inline int craft_packet(struct __ctx_buff *ctx, __be16 client_port)
{
	struct pktgen builder;
	struct tcphdr *tcph;
	void *data;

	pktgen__init(&builder, ctx);

	tcph = pktgen__push_ipv4_tcp_packet(
		&builder, (__u8 *)client_mac, (__u8 *)lb_mac, CLIENT_IP,
		FRONTEND_IP, client_port, FRONTEND_PORT);
	if (!tcph)
		return TEST_ERROR;

	data = pktgen__push_data(&builder, default_data, sizeof(default_data));
	if (!data)
		return TEST_ERROR;

	pktgen__finish(&builder);

	return 0;
}

/* Add a source range index entry: client source CIDR (optionally restricted to
 * a client source port) maps to a backend selection index.
 */
static __always_inline void add_src_range_idx(__be32 client_addr, __u8 cidr_bits,
					      __u16 sport, __u8 index)
{
	struct lb4_src_range_idx_key key = {
		.lpm_key = { 32 + cidr_bits, {} },
		.rev_nat_id = bpf_htons(REV_NAT_INDEX),
		.sport = sport,
		.addr = client_addr,
	};
	map_update_elem(&cilium_lb4_src_range_idx, &key, &index, BPF_ANY);
}

static __always_inline void del_src_range_idx(__be32 client_addr, __u8 cidr_bits,
					      __u16 sport)
{
	struct lb4_src_range_idx_key key = {
		.lpm_key = { 32 + cidr_bits, {} },
		.rev_nat_id = bpf_htons(REV_NAT_INDEX),
		.sport = sport,
		.addr = client_addr,
	};
	map_delete_elem(&cilium_lb4_src_range_idx, &key);
}

static __always_inline void setup_test(void)
{
	/* Maps are shared across the test cases of a file, so remove any
	 * entries a previous case may have left behind.
	 */
	del_src_range_idx(CLIENT_IP, 32, CLIENT_PORT);
	del_src_range_idx(CLIENT_IP, 32, 0);
	del_src_range_idx(CLIENT_CIDR, 24, 0);

	/* The high 8 bits select the source range index algorithm (4), the lower
	 * 24 bits are the session affinity timeout in seconds.
	 */
	__u32 affinity_timeout = (LB_SELECTION_SRC_RANGE_IDX << LB_ALGORITHM_SHIFT) | 100;

	__lb_v4_add_service(FRONTEND_IP, FRONTEND_PORT, IPPROTO_TCP, IPPROTO_TCP,
			    2, bpf_htons(REV_NAT_INDEX), SVC_FLAG_ROUTABLE, 0,
			    true, affinity_timeout, 0, 0);

	__lb_v4_add_backend(FRONTEND_IP, FRONTEND_PORT, 1, BACKEND_ID1, BACKEND_IP1,
			    BACKEND_PORT, IPPROTO_TCP, 0, false);
	__lb_v4_add_backend(FRONTEND_IP, FRONTEND_PORT, 2, BACKEND_ID2, BACKEND_IP2,
			    BACKEND_PORT, IPPROTO_TCP, 0, false);
}

static __always_inline int check_packet(const struct __ctx_buff *ctx,
					__u32 expected_backend)
{
	void *data, *data_end;
	__u32 *status_code;
	struct tcphdr *l4;
	struct iphdr *l3;

	test_init();

	data = (void *)(long)ctx_data(ctx);
	data_end = (void *)(long)ctx->data_end;

	if (data + 2 * sizeof(__u32) + sizeof(struct ethhdr) +
	    sizeof(struct iphdr) + sizeof(struct tcphdr) > data_end)
		test_fatal("packet out of bounds");

	/* The framework prepends the setup status, and the punted packet carries
	 * the XFER_PKT_NO_SVC marker at its start.
	 */
	status_code = data;
	assert(*status_code == CTX_ACT_OK);
	assert(*(__u32 *)(data + sizeof(__u32)) == XFER_PKT_NO_SVC);

	l3 = data + 2 * sizeof(__u32) + sizeof(struct ethhdr);
	if (__bpf_ntohl(l3->daddr) != expected_backend)
		test_fatal("dst IP hasn't been NATed to the expected backend %lx (got %lx)",
			   expected_backend, __bpf_ntohl(l3->daddr));

	l4 = (void *)l3 + sizeof(struct iphdr);
	if ((void *)l4 + sizeof(struct tcphdr) > data_end)
		test_fatal("l4 out of bounds");

	if (l4->dest != BACKEND_PORT)
		test_fatal("dst port hasn't been NATed to the backend port");

	test_finish();
}

/* An entry restricted to the client source port selects the backend
 * index % backend-count.
 */
PKTGEN("xdp", "xdp_src_range_idx_exact")
int xdp_src_range_idx_exact_pktgen(struct __ctx_buff *ctx)
{
	return craft_packet(ctx, CLIENT_PORT);
}

SETUP("xdp", "xdp_src_range_idx_exact")
int xdp_src_range_idx_exact_setup(struct __ctx_buff *ctx)
{
	setup_test();
	add_src_range_idx(CLIENT_IP, 32, CLIENT_PORT, 1);

	return xdp_receive_packet(ctx);
}

CHECK("xdp", "xdp_src_range_idx_exact")
int xdp_src_range_idx_exact_check(const struct __ctx_buff *ctx)
{
	return check_packet(ctx, __bpf_ntohl(BACKEND_IP2));
}

/* A wildcard (port 0) entry is matched by the fallback lookup when no
 * port-specific entry exists.
 */
PKTGEN("xdp", "xdp_src_range_idx_wildcard")
int xdp_src_range_idx_wildcard_pktgen(struct __ctx_buff *ctx)
{
	return craft_packet(ctx, CLIENT_PORT2);
}

SETUP("xdp", "xdp_src_range_idx_wildcard")
int xdp_src_range_idx_wildcard_setup(struct __ctx_buff *ctx)
{
	setup_test();
	add_src_range_idx(CLIENT_IP, 32, 0, 0);

	return xdp_receive_packet(ctx);
}

CHECK("xdp", "xdp_src_range_idx_wildcard")
int xdp_src_range_idx_wildcard_check(const struct __ctx_buff *ctx)
{
	return check_packet(ctx, __bpf_ntohl(BACKEND_IP1));
}

/* A source CIDR entry with a shorter prefix matches clients inside the CIDR
 * via longest-prefix-match.
 */
PKTGEN("xdp", "xdp_src_range_idx_cidr")
int xdp_src_range_idx_cidr_pktgen(struct __ctx_buff *ctx)
{
	return craft_packet(ctx, CLIENT_PORT3);
}

SETUP("xdp", "xdp_src_range_idx_cidr")
int xdp_src_range_idx_cidr_setup(struct __ctx_buff *ctx)
{
	setup_test();
	add_src_range_idx(CLIENT_CIDR, 24, 0, 1);

	return xdp_receive_packet(ctx);
}

CHECK("xdp", "xdp_src_range_idx_cidr")
int xdp_src_range_idx_cidr_check(const struct __ctx_buff *ctx)
{
	return check_packet(ctx, __bpf_ntohl(BACKEND_IP2));
}
