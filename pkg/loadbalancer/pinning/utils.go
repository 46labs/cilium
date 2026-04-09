package pinning

import (
	"github.com/cilium/cilium/pkg/k8s/resource"
	k8sRuntime "k8s.io/apimachinery/pkg/runtime"
)

func eventDone[T k8sRuntime.Object](event resource.Event[T], err error) error {
	event.Done(err)

	return err
}
