// SPDX-License-Identifier: (GPL-2.0-only OR BSD-2-Clause)
/* Copyright Authors of Cilium */

#include <bpf/ctx/skb.h>
#include "common.h"
#include "pktgen.h"

/* Enable code paths under test */
#define ENABLE_IPV4			1
#define ENABLE_IPV6			1
#define ENABLE_NODEPORT			1
#define ENABLE_EGRESS_GATEWAY		1
#define ENABLE_MASQUERADE_IPV4		1
#define ENABLE_MASQUERADE_IPV6		1
#define ENABLE_HOST_FIREWALL		1
#define ENCAP_IFINDEX 0

#include "lib/bpf_host.h"

#include "lib/egressgw.h"
#include "lib/endpoint.h"
#include "lib/ipcache.h"

#define SERVICE_TOS	0xb8
#define OTHER_TOS	0x10

static __always_inline void egressgw_tos_common_setup(struct __ctx_buff *ctx)
{
	ipcache_v4_add_world_entry();
	create_ct_entry(ctx, client_port(TEST_REDIRECT));
	ipcache_v4_add_entry(EGRESS_IP, 0, HOST_ID, 0, 0);
}

static __always_inline int egressgw_tos_setup(struct __ctx_buff *ctx,
					      bool with_fallback)
{
	egressgw_tos_common_setup(ctx);

	add_egressgw_policy_entry_tos(CLIENT_IP, EXTERNAL_SVC_IP, SERVICE_TOS, GATEWAY_NODE_IP,
				      EGRESS_IP);
	if (with_fallback)
		add_egressgw_policy_entry_tos(CLIENT_IP, EXTERNAL_SVC_IP, 0, GATEWAY_NODE_IP,
					      EGRESS_IP);

	return netdev_send_packet(ctx);
}

static __always_inline void egressgw_tos_cleanup(bool with_fallback)
{
	if (with_fallback)
		del_egressgw_policy_entry_tos(CLIENT_IP, EXTERNAL_SVC_IP, 0);
	del_egressgw_policy_entry_tos(CLIENT_IP, EXTERNAL_SVC_IP, SERVICE_TOS);
}

static __always_inline int egressgw_tos_pktgen(struct __ctx_buff *ctx, __u8 tos)
{
	return egressgw_pktgen(ctx, (struct egressgw_test_ctx) {
			.test = TEST_REDIRECT,
			.tos = tos,
		});
}

/* Test that a packet carrying the TOS pinned by an egress gateway policy on
 * the to-netdev program gets redirected to the gateway node.
 */
PKTGEN("tc", "tc_egressgw_tos_pinned")
int egressgw_tos_pinned_pktgen(struct __ctx_buff *ctx)
{
	return egressgw_tos_pktgen(ctx, SERVICE_TOS);
}

SETUP("tc", "tc_egressgw_tos_pinned")
int egressgw_tos_pinned_setup(struct __ctx_buff *ctx)
{
	return egressgw_tos_setup(ctx, true);
}

CHECK("tc", "tc_egressgw_tos_pinned")
int egressgw_tos_pinned_check(const struct __ctx_buff *ctx)
{
	int ret = egressgw_status_check(ctx, (struct egressgw_test_ctx) {
			.status_code = TC_ACT_REDIRECT,
	});

	egressgw_tos_cleanup(true);

	return ret;
}

/* Test that a packet carrying a TOS that does not match the pinned one falls
 * back to the entry with a pinned TOS of 0 and gets redirected.
 */
PKTGEN("tc", "tc_egressgw_tos_fallback")
int egressgw_tos_fallback_pktgen(struct __ctx_buff *ctx)
{
	return egressgw_tos_pktgen(ctx, OTHER_TOS);
}

SETUP("tc", "tc_egressgw_tos_fallback")
int egressgw_tos_fallback_setup(struct __ctx_buff *ctx)
{
	return egressgw_tos_setup(ctx, true);
}

CHECK("tc", "tc_egressgw_tos_fallback")
int egressgw_tos_fallback_check(const struct __ctx_buff *ctx)
{
	int ret = egressgw_status_check(ctx, (struct egressgw_test_ctx) {
			.status_code = TC_ACT_REDIRECT,
	});

	egressgw_tos_cleanup(true);

	return ret;
}

/* Test that without an entry with a pinned TOS of 0, a packet carrying a TOS
 * that does not match the pinned one is not redirected.
 */
PKTGEN("tc", "tc_egressgw_tos_no_fallback")
int egressgw_tos_no_fallback_pktgen(struct __ctx_buff *ctx)
{
	return egressgw_tos_pktgen(ctx, OTHER_TOS);
}

SETUP("tc", "tc_egressgw_tos_no_fallback")
int egressgw_tos_no_fallback_setup(struct __ctx_buff *ctx)
{
	return egressgw_tos_setup(ctx, false);
}

CHECK("tc", "tc_egressgw_tos_no_fallback")
int egressgw_tos_no_fallback_check(const struct __ctx_buff *ctx)
{
	int ret = egressgw_status_check(ctx, (struct egressgw_test_ctx) {
			.status_code = TC_ACT_OK,
	});

	egressgw_tos_cleanup(false);

	return ret;
}
