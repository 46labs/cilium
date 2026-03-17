#include <bpf/ctx/xdp.h>

#include <linux/if_ether.h>
#include <linux/in.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <linux/bpf.h>

#include "lib/endian.h"
#include "lib/overloadable_xdp.h"

// BPF_LICENSE("GPL");
