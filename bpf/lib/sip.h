/* SPDX-License-Identifier: (GPL-2.0-only OR BSD-2-Clause) */
/* Copyright 46labs LLC */

#pragma once

#include <linux/udp.h>

#ifdef ENABLE_SIP_INSPECTION

#define NOT_FOUND 0

/* sip_inspect() uses direct packet access. TC skbs received after tunnel
 * decapsulation may keep the SIP payload outside the linear head, even though
 * skb_load_bytes() can still read the SIP method. Pull only the parser window
 * that can actually be inspected: header lines starting in the first 1000
 * bytes, the 9-byte header name and the value hashed after it.
 */
#define SIP_L3_HEADERS_LEN (sizeof(struct ethhdr) + sizeof(struct iphdr))
#define SIP_PAYLOAD_OFFSET (SIP_L3_HEADERS_LEN + sizeof(struct udphdr))
#define SIP_METHOD_LEN 10
#define SIP_PARSE_WINDOW (SIP_PAYLOAD_OFFSET + 1000 + 9 + 68)
/* 8-byte words scanned for line starts: the first 1000 payload bytes. */
#define SIP_SCAN_WORDS 125

static inline __u8 is_sip(const char *cur, const char *data_end)
{
	static const __u64 resp = 0x20302e322f706973; // "sip/2.0 "

	// 8 bytes match
	static const __u64 reg = 0x7265747369676572;	    // "register"
	static const __u64 subscribe1 = 0x6269726373627573; // "subscrib"

	// 8 bytes with space
	static const __u64 options = 0x20736e6f6974706f; // "options "
	static const __u64 message = 0x206567617373656d; // "message "
	static const __u64 publish = 0x206873696c627570; // "publish "

	// 7 bytes with space
	static const __u64 invite = 0x0020657469766e69; // "invite "
	static const __u64 cancel = 0x00206c65636e6163; // "cancel "
	static const __u64 update = 0x0020657461647075; // "update "
	static const __u64 notify = 0x0020796669746f6e; // "notify "

	// 6 bytes match
	static const __u64 prack = 0x0000206b63617270; // "prack "
	static const __u64 refer = 0x0000207265666572; // "refer "

	// 5 bytes match
	static const __u64 info = 0x000000206f666e69; // "info "

	// 4 bytes match
	static const __u64 ack = 0x00000000206b6361; // "ack "
	static const __u64 bye = 0x0000000020657962; // "bye "

	// remainders
	static const __u16 subscribe2 = 0x2065; // "e "

	if (cur + sizeof(__u64) + sizeof(__u16) > data_end)
		return 0;

	__u64 v = *(__u64 *)cur;
	__u16 r = *(__u16 *)(cur + 8);

	if ((v | 0x202020) == resp)
		return 1;

	__u64 v_masked8 = v | 0x2020202020202020;
	__u16 r_masked1 = r | 0x0020;
	if (v_masked8 == reg)
		return 1;

	if (v_masked8 == subscribe1 && r_masked1 == subscribe2)
		return 1;

	__u64 v_masked7 = v | 0x0020202020202020;
	if (v_masked7 == options || v_masked7 == message || v_masked7 == publish)
		return 1;

	__u64 v_masked6 = (v | 0x0000202020202020) & 0x00FFFFFFFFFFFFFF;
	if (v_masked6 == invite || v_masked6 == cancel || v_masked6 == update ||
	    v_masked6 == notify)
		return 1;

	__u64 v_masked5 = (v | 0x0000002020202020) & 0x0000FFFFFFFFFFFF;
	if (v_masked5 == prack || v_masked5 == refer)
		return 1;

	__u64 v_masked4 = (v | 0x0000000020202020) & 0x000000FFFFFFFFFF;
	if (v_masked4 == info)
		return 1;

	__u64 v_masked3 = (v | 0x0000000000202020) & 0x00000000FFFFFFFF;
	if (v_masked3 == ack || v_masked3 == bye)
		return 1;

	return 0;
}

/* FNV-1a of the Call-ID value at off: one optional leading whitespace byte
 * skipped, at most 64 bytes, and the value must end in CRLF inside the packet.
 * A global function, so the verifier checks it once rather than once per scan
 * position in the caller.
 */
__noinline __weak __u32 sip_hash_call_id(struct __ctx_buff *ctx, __u32 off)
{
	void *data = (void *)(long)ctx->data;
	void *data_end = (void *)(long)ctx->data_end;
	__u32 hash = 0x811c9dc5;
	void *cur;

	if (off > SIP_PARSE_WINDOW)
		return NOT_FOUND;
	cur = data + off;
	if (cur + 1 > data_end)
		return NOT_FOUND;
	if (*(__u8 *)cur == ' ' || *(__u8 *)cur == '\t')
		cur++;

	for (int i = 0; i < 64; i++) {
		if (cur + i + 2 > data_end)
			return NOT_FOUND;
		if (*(__u8 *)(cur + i) == '\r' && *(__u8 *)(cur + i + 1) == '\n')
			return i ? hash : NOT_FOUND;
		hash ^= *(__u8 *)(cur + i);
		hash *= 0x01000193;
	}
	return hash;
}

__noinline __weak __u32 sip_inspect(struct __ctx_buff *ctx)
{
	static const __u64 call_id1 = 0x3a64692d6c6c6163ULL; // "call-id:"

	void *data, *data_end;
	struct ethhdr *eth = NULL;
	struct iphdr *iph = NULL;
	__u64 method[2] = {};
	__u32 pull_len;

	data = (void *)(long)ctx->data;
	data_end = (void *)(long)ctx->data_end;

	if (data + SIP_L3_HEADERS_LEN > data_end)
		return NOT_FOUND;

	eth = data;
	if ((void *)(eth + 1) > data_end)
		return NOT_FOUND;

	if (eth->h_proto != bpf_htons(ETH_P_IP))
		return NOT_FOUND;

	iph = (void *)(eth + 1);

	if (iph->protocol != IPPROTO_UDP)
		return NOT_FOUND;

	if ((void *)(iph + 1) > data_end)
		return NOT_FOUND;

	void *cur = data + SIP_PAYLOAD_OFFSET;
	if (cur + SIP_METHOD_LEN > data_end) {
		/* skb_load_bytes() can read non-linear data without invalidating
		 * packet pointers. Only linearize the larger parser window after
		 * confirming that this is actually SIP.
		 */
		if (ctx_load_bytes(ctx, SIP_PAYLOAD_OFFSET, method,
				   SIP_METHOD_LEN) < 0 ||
		    !is_sip((char *)method, (char *)method + SIP_METHOD_LEN))
			return NOT_FOUND;
	} else if (!is_sip(cur, data_end)) {
		return NOT_FOUND;
	}

	/* The scan below uses direct access too. A decapsulated skb may have only
	 * its headers linear even though the complete SIP message is present.
	 */
	pull_len = (__u32)ctx_full_len(ctx);
	if (pull_len > SIP_PARSE_WINDOW)
		pull_len = SIP_PARSE_WINDOW;
	if (data + pull_len > data_end) {
		if (ctx_pull_data(ctx, pull_len) < 0)
			return NOT_FOUND;

		data = (void *)(long)ctx->data;
		data_end = (void *)(long)ctx->data_end;

		if (data + SIP_PAYLOAD_OFFSET + SIP_METHOD_LEN > data_end)
			return NOT_FOUND;
	}

	/* Fixed 8-byte stride, so word loads do not depend on each other. A word
	 * holding a LF tests the header name at the start of the next line. Line
	 * starts stay inside the pulled parser window.
	 */
	for (int i = 0; i < SIP_SCAN_WORDS; i++) {
		__u32 base = SIP_PAYLOAD_OFFSET + i * 8;

		cur = data + base;
		if (ctx_no_room(cur + 8, data_end))
			return NOT_FOUND;

		__u64 x = *(__u64 *)cur ^ 0x0a0a0a0a0a0a0a0aULL;
		__u64 t = (x - 0x0101010101010101ULL) & ~x & 0x8080808080808080ULL;

		for (int j = 0; j < 3 && t; j++) {
			__u64 low = t & -t;
			/* 1..8: bytes from cur to the start of the next line. */
			__u32 adv = ((((low >> 7) * 0x0102030405060708ULL) >> 56) & 0xf);
			__u32 off = base + adv;
			void *line = data + off;

			t &= t - 1;
			/* The borrow can flag the byte above a LF; check it. */
			if (ctx_no_room(line + 2, data_end))
				return NOT_FOUND;
			if (*(__u8 *)(line - 1) != '\n')
				continue;

			__u16 h = *(__u16 *)line | 0x2020;

			if (h == 0x3a69)
				return sip_hash_call_id(ctx, off + 2);
			if (h != 0x6163 || ctx_no_room(line + 8, data_end))
				continue;
			if ((*(__u64 *)line | 0x0020200020202020ULL) == call_id1)
				return sip_hash_call_id(ctx, off + 8);
		}
	}
	return NOT_FOUND;
}

#endif /* ENABLE_SIP_INSPECTION */
