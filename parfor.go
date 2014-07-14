// Copyright 2013 Brendan Tracey. All rights reserved.
// Use of this code is governed by a BSD-style
// license that can be found in the LICENSE file

package nnet

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// ParallelFor computes the function f in parallel
func ParallelFor(n, grain int, f func(start, end int)) {
	P := runtime.GOMAXPROCS(0)
	idx := uint64(0)
	var wg sync.WaitGroup
	wg.Add(P)
	for p := 0; p < P; p++ {
		go func() {
			for {
				start := int(atomic.AddUint64(&idx, uint64(grain))) - grain
				if start >= n {
					break
				}
				end := start + grain
				if end > n {
					end = n
				}
				f(start, end)
			}
			wg.Done()
		}()
	}
	wg.Wait()
}
