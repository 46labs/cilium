// SPDX-License-Identifier: Apache-2.0
// Copyright Authors of Cilium

package loadbalancer

import (
	"fmt"
	"maps"
	"math"
	"net/netip"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/cilium/statedb"
	"github.com/cilium/statedb/index"

	"github.com/cilium/cilium/pkg/annotation"
	"github.com/cilium/cilium/pkg/labels"
	"github.com/cilium/cilium/pkg/source"
	"github.com/cilium/cilium/pkg/time"
)

// Service defines the common properties for a load-balancing service. Associated with a
// service are a set of frontends that receive the traffic, and a set of backends to which
// the traffic is directed. A single frontend can map to a partial subset of backends depending
// on its properties.
type Service struct {
	// Name is the fully qualified service name, e.g. (<cluster>/)<namespace>/<name>.
	Name ServiceName

	// Source is the data source from which this service was ingested from.
	Source source.Source

	// Labels associated with the service.
	Labels labels.Labels

	// Annotations associated with this service.
	Annotations map[string]string

	// Selector specifies which pods should be associated with this service. If
	// this is empty the backends associated to this service are managed externally
	// and not by Kubernetes.
	Selector map[string]string

	// NatPolicy defines whether we need NAT46/64 translation for backends.
	NatPolicy SVCNatPolicy

	// ExtTrafficPolicy controls how backends are selected for North-South traffic.
	// If set to "Local", only node-local backends are chosen.
	ExtTrafficPolicy SVCTrafficPolicy

	// IntTrafficPolicy controls how backends are selected for East-West traffic.
	// If set to "Local", only node-local backends are chosen.
	IntTrafficPolicy SVCTrafficPolicy

	// ForwardingMode controls whether DSR or SNAT should be used for the dispatch
	// to the backend. If undefined the default mode is used (--bpf-lb-mode).
	ForwardingMode SVCForwardingMode

	// SessionAffinity if true will enable the client IP based session affinity.
	SessionAffinity bool

	// SessionAffinityTimeout is the duration of inactivity before the session
	// affinity is cleared for a specific client IP.
	SessionAffinityTimeout time.Duration

	// LoadBalancerClass if set specifies the load-balancer class to be used
	// for a LoadBalancer service. If unset the default implementation is used.
	LoadBalancerClass *string

	// ProxyRedirect if non-nil redirects the traffic going to the frontends
	// towards a locally running proxy.
	ProxyRedirect *ProxyRedirect

	// HealthCheckNodePort defines on which port the node runs a HTTP health
	// check server which may be used by external loadbalancers to determine
	// if a node has local backends. This will only have effect if both
	// LoadBalancerIPs is not empty and ExtTrafficPolicy is SVCTrafficPolicyLocal.
	HealthCheckNodePort uint16

	// LoopbackHostPort defines that HostPort frontends for this service should
	// only be exposed internally to the node.
	LoopbackHostPort bool

	// SourceRanges if non-empty will restrict access to the service to the specified
	// client addresses.
	SourceRanges []netip.Prefix

	// SourceRangeIndexes if non-empty maps client source CIDRs (optionally
	// restricted to a client source port) to a trunk group index. Entries
	// sharing an Index form a trunk group; the backend selected for a
	// matching client is whichever backend's pod carries a
	// PodSourceRangeGroup label matching that Index. If no backend currently
	// claims the index, matching traffic is dropped.
	SourceRangeIndexes []SourceRangeIndexEntry

	// PortNames maps a port name to a port number.
	PortNames map[string]uint16

	// TrafficDistribution if not default will influence how backends are chosen for
	// frontends associated with this service.
	TrafficDistribution TrafficDistribution
}

type SourceRangeIndexEntry struct {
	// Index is the trunk group index shared by all client source CIDRs of the
	// group; see SourceRangeIndexes for how it selects a backend.
	Index uint8

	// Prefix is the client source CIDR this entry applies to.
	Prefix netip.Prefix

	// Port if non-zero restricts this entry to the given client source port.
	// A zero port matches any client source port.
	Port uint16
}

func (e SourceRangeIndexEntry) String() string {
	if e.Port != 0 {
		return fmt.Sprintf("%d:%s:%d", e.Index, e.Prefix, e.Port)
	}
	return fmt.Sprintf("%d:%s", e.Index, e.Prefix)
}

// ParseSourceRangeIndexes parses the value of the
// ServiceSourceRangeIndex annotation into source range index entries.
//
// The value is a semicolon-separated list of trunk groups. Each group is a
// comma-separated list of "CIDR[:port]" entries, where each entry carries its
// own optional client source port. All entries of a group share the group
// index, which is the position of the group in the list.
func ParseSourceRangeIndexes(value string) ([]SourceRangeIndexEntry, error) {
	var result []SourceRangeIndexEntry

	groupIndex := 0
	for _, group := range strings.Split(value, ";") {
		group = strings.TrimSpace(group)
		if group == "" {
			continue
		}
		if groupIndex > math.MaxUint8 {
			return nil, fmt.Errorf("too many source range index groups: %d", groupIndex)
		}

		for _, entry := range strings.Split(group, ",") {
			entry = strings.TrimSpace(entry)
			if entry == "" {
				return nil, fmt.Errorf("invalid source range index group %q: empty entry", group)
			}

			addrPart, portPart, err := splitSourceRangeAddrPort(entry)
			if err != nil {
				return nil, err
			}

			var port uint16
			if portPart != "" {
				p, err := strconv.ParseUint(portPart, 10, 16)
				if err != nil {
					return nil, fmt.Errorf("invalid source range index port %q: %w", portPart, err)
				}
				port = uint16(p)
			}

			prefix, err := parseSourceRangePrefix(addrPart)
			if err != nil {
				return nil, err
			}

			result = append(result, SourceRangeIndexEntry{
				Index:  uint8(groupIndex),
				Prefix: prefix,
				Port:   port,
			})
		}
		groupIndex++
	}

	return result, nil
}

// splitSourceRangeAddrPort splits a "CIDR[:port]" entry into its address part
// and, if present, its port part. Bracketed IPv6 addresses, e.g.
// "[fd00::1]/128:5060", are supported.
func splitSourceRangeAddrPort(entry string) (addrPart, portPart string, err error) {
	addrPart, portPart = entry, ""

	// A bracketed IPv6 address, e.g. "[fd00::1]/128:5060".
	if strings.HasPrefix(entry, "[") {
		end := strings.Index(entry, "]")
		if end < 0 {
			return "", "", fmt.Errorf("invalid IPv6 address in %q: missing closing bracket", entry)
		}
		rest := entry[end+1:]
		addrPart = entry[1:end] + rest
		if i := strings.LastIndex(rest, ":"); i >= 0 {
			addrPart = entry[1:end] + rest[:i]
			portPart = rest[i+1:]
		}
		return addrPart, portPart, nil
	}

	if i := strings.LastIndex(entry, ":"); i >= 0 {
		// Only treat the suffix as a port if it parses as one, otherwise
		// an unbracketed IPv6 address is assumed, e.g. "fd00::1/128".
		if _, parseErr := strconv.ParseUint(entry[i+1:], 10, 16); parseErr == nil {
			addrPart = entry[:i]
			portPart = entry[i+1:]
		}
	}
	return addrPart, portPart, nil
}

// parseSourceRangePrefix parses a CIDR, allowing a bare IP which defaults to a
// full-length prefix.
func parseSourceRangePrefix(addrPart string) (netip.Prefix, error) {
	prefix, err := netip.ParsePrefix(addrPart)
	if err != nil {
		// Allow a bare IP, defaulting to a full-length prefix.
		if ip, ipErr := netip.ParseAddr(addrPart); ipErr == nil {
			return netip.PrefixFrom(ip, ip.BitLen()), nil
		}
		return netip.Prefix{}, fmt.Errorf("invalid source range index entry %q: %w", addrPart, err)
	}
	return prefix.Masked(), nil
}

type TrafficDistribution string

const (
	// TrafficDistributionDefault will ignore any topology aware hints for choosing the backends.
	TrafficDistributionDefault = TrafficDistribution("")

	// TrafficDistributionPreferClose Indicates preference for routing traffic to topologically close backends,
	// that is to backends that are in the same zone.
	TrafficDistributionPreferClose = TrafficDistribution("PreferClose")
)

func (svc *Service) GetLBAlgorithmAnnotation() SVCLoadBalancingAlgorithm {
	return ToSVCLoadBalancingAlgorithm(svc.Annotations[annotation.ServiceLoadBalancingAlgorithm])
}

func (svc *Service) GetSipInspect() bool {
	_, ret := svc.Annotations[annotation.ServiceSipInspect]
	return ret
}

// GetTOS returns the TOS byte value configured for the service via the
// service.cilium.io/tos annotation. The value may be specified either in
// decimal or hexadecimal (e.g. "0xb8") notation. If the annotation is
// absent or invalid, the second return value is false and the TOS of
// packets hitting the service is left untouched.
func (svc *Service) GetTOS() (uint8, bool) {
	value, ok := svc.Annotations[annotation.ServiceTOS]
	if !ok {
		return 0, false
	}

	tos, err := strconv.ParseUint(value, 0, 8)
	if err != nil {
		return 0, false
	}

	return uint8(tos), true
}

func (svc *Service) GetProxyDelegation() SVCProxyDelegation {
	if value, ok := annotation.Get(svc, annotation.ServiceProxyDelegation); ok {
		tmp := SVCProxyDelegation(strings.ToLower(value))
		if tmp == SVCProxyDelegationDelegateIfLocal {
			return tmp
		}
	}
	return SVCProxyDelegationNone
}

func (svc *Service) GetSourceRangesPolicy() SVCSourceRangesPolicy {
	if value, ok := annotation.Get(svc, annotation.ServiceSourceRangesPolicy); ok {
		if SVCSourceRangesPolicy(strings.ToLower(value)) == SVCSourceRangesPolicyDeny {
			return SVCSourceRangesPolicyDeny
		}
	}
	return SVCSourceRangesPolicyAllow
}

func (svc *Service) GetSourceRangesEnabled(svcType SVCType, lbSourceRangeAllTypes bool) bool {
	if lbSourceRangeAllTypes {
		return len(svc.SourceRanges) > 0
	} else {
		return len(svc.SourceRanges) > 0 && svcType == SVCTypeLoadBalancer
	}
}

func (svc *Service) GetAnnotations() map[string]string {
	return svc.Annotations
}

type ProxyRedirect struct {
	ProxyPort uint16

	// Ports if non-empty will only redirect a frontend with a matching port.
	Ports []uint16
}

func (pr *ProxyRedirect) Redirects(port uint16) bool {
	if pr == nil {
		return false
	}
	return len(pr.Ports) == 0 || slices.Contains(pr.Ports, port)
}

func (pr *ProxyRedirect) Equal(other *ProxyRedirect) bool {
	switch {
	case pr == nil && other == nil:
		return true
	case pr != nil && other != nil:
		return pr.ProxyPort == other.ProxyPort && slices.Equal(pr.Ports, other.Ports)
	default:
		return false
	}
}

func (pr *ProxyRedirect) String() string {
	if pr == nil {
		return ""
	}
	if len(pr.Ports) > 0 {
		return fmt.Sprintf("%d (ports: %v)", pr.ProxyPort, pr.Ports)
	}
	return strconv.FormatUint(uint64(pr.ProxyPort), 10)
}

// Clone returns a shallow clone of the service, e.g. for updating a service with UpsertService. Fields that are references
// (e.g. Labels or Annotations) must be further cloned if mutated.
func (svc *Service) Clone() *Service {
	svc2 := *svc
	return &svc2
}

func (svc *Service) TableHeader() []string {
	// NOTE: Annotations and labels are not shown here as they're rarely interesting for debugging.
	// They are still available for inspection via "cilium-dbg statedb dump".
	return []string{
		"Name",
		"Source",
		"PortNames",
		"TrafficPolicy",
		"Flags",
	}
}

func (svc *Service) TableRow() []string {
	var trafficPolicy string
	if svc.ExtTrafficPolicy == svc.IntTrafficPolicy {
		trafficPolicy = string(svc.ExtTrafficPolicy)
	} else {
		trafficPolicy = fmt.Sprintf("Ext=%s, Int=%s", svc.ExtTrafficPolicy, svc.IntTrafficPolicy)
	}

	// Collapse the more rarely set fields into a single "Flags" column
	var flags []string

	if svc.SessionAffinity {
		flags = append(flags, "SessionAffinity="+svc.SessionAffinityTimeout.String())
	}

	if len(svc.SourceRanges) > 0 {
		cidrs := svc.SourceRanges
		ss := make([]string, len(cidrs))
		for i := range cidrs {
			ss[i] = cidrs[i].String()
		}
		flags = append(flags, "SourceRanges="+strings.Join(ss, ", "))
	}

	if len(svc.SourceRangeIndexes) > 0 {
		entries := make([]string, len(svc.SourceRangeIndexes))
		for i := range svc.SourceRangeIndexes {
			entries[i] = svc.SourceRangeIndexes[i].String()
		}
		flags = append(flags, "SourceRangeIndexes="+strings.Join(entries, ", "))
	}

	if p := svc.GetSourceRangesPolicy(); p == SVCSourceRangesPolicyDeny {
		flags = append(flags, "SourceRangesPolicy=deny")
	}

	if svc.ProxyRedirect != nil {
		flags = append(flags, "ProxyRedirect="+svc.ProxyRedirect.String())
	}

	if svc.HealthCheckNodePort != 0 {
		flags = append(flags, fmt.Sprintf("HealthCheckNodePort=%d", svc.HealthCheckNodePort))
	}

	if svc.LoopbackHostPort {
		flags = append(flags, "LoopbackHostPort="+strconv.FormatBool(svc.LoopbackHostPort))
	}

	if alg := svc.GetLBAlgorithmAnnotation(); alg != SVCLoadBalancingAlgorithmUndef {
		flags = append(flags, "ExplicitLBAlgorithm="+alg.String())
	}

	if sipInspect := svc.GetSipInspect(); sipInspect {
		flags = append(flags, "SipInspect")
	}

	if svc.ForwardingMode != SVCForwardingModeUndef {
		flags = append(flags, "ForwardingMode="+string(svc.ForwardingMode))
	}

	if svc.TrafficDistribution != TrafficDistributionDefault {
		flags = append(flags, "TrafficDistribution="+string(svc.TrafficDistribution))
	}

	if svc.LoadBalancerClass != nil {
		flags = append(flags, "LoadBalancerClass="+*svc.LoadBalancerClass)
	}

	sort.Strings(flags)

	return []string{
		svc.Name.String(),
		string(svc.Source),
		svc.showPortNames(),
		trafficPolicy,
		strings.Join(flags, ", "),
	}
}

func (svc *Service) showPortNames() string {
	var b strings.Builder
	n := len(svc.PortNames)
	for _, name := range slices.Sorted(maps.Keys(svc.PortNames)) {
		fmt.Fprintf(&b, "%s=%d", name, svc.PortNames[name])
		n--
		if n > 0 {
			b.WriteString(", ")
		}

	}
	return b.String()
}

var (
	serviceNameIndex = statedb.Index[*Service, ServiceName]{
		Name: "name",
		FromObject: func(obj *Service) index.KeySet {
			return index.NewKeySet(obj.Name.Key())
		},
		FromKey:    ServiceName.Key,
		FromString: index.FromString,
		Unique:     true,
	}

	ServiceByName = serviceNameIndex.Query
)

const (
	ServiceTableName = "services"
)

func NewServicesTable(cfg Config, db *statedb.DB) (statedb.RWTable[*Service], error) {
	return statedb.NewTable(
		db,
		ServiceTableName,
		serviceNameIndex,
	)
}
