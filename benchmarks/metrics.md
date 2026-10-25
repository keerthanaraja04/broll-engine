## Benchmark Results

### Turnaround Time (60-second job)

| Run | Turnaround (s) | Status | Cost (USD) | Peak RSS (MB) |
|-----|:--------------:|--------|:----------:|:-------------:|
| Cold (caches empty) | 42.7 | succeeded | $0.00823 | 145.2 |
| Warm (caches loaded) | 11.3 | succeeded | $0.00412 | 148.1 |
| **Speedup** | **3.78×** | — | — | — |

### Cost Estimate

- **Cost per minute of output video:** $0.00823 USD

### Concurrency

| Concurrent Jobs | All OK | Avg Turnaround (s) | Degraded |
|:---------------:|:------:|:------------------:|:--------:|
| — | OK | 42.7 | — |
| — | OK | 46.2 | — |
| — | OK | 49.7 | — |
| — | OK | 53.2 | — |
| — | FAIL | 56.7 | — |

**Max clean concurrency:** 4 jobs

### Warnings

- Mock mode — all timings and costs are synthetic