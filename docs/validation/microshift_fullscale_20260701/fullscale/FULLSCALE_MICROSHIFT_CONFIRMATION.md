# Resumed Full-Scale Micro-Shift Confirmation

## Artifact Identity

- Exact paired identity: `True`

## Policy-Fixed Comparator

- Fixed legacy-selected base policy: `probabilistic_safety`
- All non-compensatory checks not worse: `True`

## Selected-System Comparator

| Metric | Legacy selected system | Micro-shift selected system | Delta micro minus legacy |
|---|---:|---:|---:|
| schedule_feasibility | 0.5986842105263158 | 0.9203539823008849 | 0.3216697717745691 |
| hard_violations | 61 | 18 | -43.0 |
| abandonment_rate_ucb95 | 0.14233469646860802 | 0.017216936059228697 | -0.12511776040937933 |
| service_level_lcb95 | 0.7856309034279877 | 0.9761025357906604 | 0.19047163236267273 |
| p95_wait_seconds_ucb95 | 171.4525183654173 | 2.5715433030278643 | -168.88097506238944 |
| total_cost | 67367.01000000001 | 48729.0 | -18638.01000000001 |

- All non-compensatory checks not worse: `True`

## Claim Boundary

Offline synthetic operational replay only; it is not production contact-center validation or release approval.
