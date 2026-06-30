# Release Qualification Report — support-capacity-reliability 1.4.1rc2

## 1. Final verdict

**CONDITIONALLY QUALIFIED**

- **Local status:** PASS — Linux x86_64, Python 3.13.5, CPU canonical path, clean Q2 source install and Q3 rebuilt-wheel install.
- **GitHub status:** GitHub configuration validated only; GitHub-hosted run **NOT EXECUTED**.
- **GitHub evidence level:** Q0 workflow inspection + Q2 local workflow-equivalent; Q4 unavailable.
- **Qualified commit:** NOT_AVAILABLE — input archive contained no `.git` metadata.
- **Qualified version:** `1.4.1rc2`
- **Qualified source fingerprint:** `6cfdc83de28d7d80640540b682c65917a37932b25e02c4581355a4b8b12be848`
- **Qualified artifacts:** `support_capacity_reliability-1.4.1rc2-py3-none-any.whl`, `support_capacity_reliability-1.4.1rc2.tar.gz`; outer release ZIP checksum is recorded in the non-self-referential sidecar `release_bundle_manifest.json`.

The verdict is conditional only because exact-commit GitHub-hosted, Windows, Docker, and GPU/Chronos execution were unavailable. No unexecuted path is marked PASS.

## 2. Scope of qualification

Executed: repository and package audit, clean non-editable source install, exact dependency consistency, two consecutive full suites, wheel/sdist build and content inspection, rebuilt-wheel-only install, CLI contracts, canonical and negative-gate E2E, artifact publication/reload, real-socket API lifecycle and invalid-request tests, concurrent writer rejection, corruption tests, and API performance observation.

Not executed: GitHub-hosted runners, remote push/release/workflow dispatch, Windows runtime, Docker runtime, CUDA/GPU/Chronos-2, and external vulnerability-database audit.

## 3. Environment

| Item | Value |
|---|---|
| OS | `Linux-4.4.0-x86_64-with-glibc2.41` |
| Runtime | Python 3.13.5 |
| Architecture | x86_64 |
| CPU | 56 logical CPUs reported |
| GPU/CUDA | NOT_EXECUTED; `nvidia-smi` unavailable |
| Package manager | pip + venv + `pyproject.toml` |
| Exact tested constraints | `constraints/qualification-py313.txt` |
| Docker | NOT_EXECUTED; executable unavailable |
| Q2 path | space/Unicode clean source environment |
| Q3 path | separate built-wheel-only space/Unicode environment |

## 4. Problems found and fixed

| Severity | Problem | Root cause | Fix | Regression test | Result |
|---|---|---|---|---|---|
| High | CLI가 JSON을 출력한 뒤 native-library shutdown에서 종료되지 않을 수 있음 | 장시간/solver pipeline을 public interpreter에서 직접 실행하거나 public CLI 전체에 os._exit 우회를 적용하면 lifecycle 및 cleanup contract가 깨짐 | pipeline/doctor/model-bundle probe만 bounded isolated worker에서 실행하고 public CLI는 정상 종료; timeout·partial-output·busy exit contract 추가 | test_process_utils.py, test_runtime.py, test_doctor.py, test_cli.py 및 실제 CLI 반복 실행 | PASS |
| High | Base install이 사실상 PyTorch에 숨은 의존 | pipeline/factory의 eager torch import와 base dependencies의 torch 선언 | lazy import와 torch/chronos/full optional extras; CPU canonical path는 LightGBM+RCWE 유지 | test_optional_dependencies.py, test_torch_model.py, clean Q2 base install/E2E | PASS |
| High | Core report path에서 tabulate 미선언 | DataFrame.to_markdown() 사용과 runtime manifest 불일치 | tabulate>=0.9,<0.10를 runtime dependency와 doctor에 추가 | clean Q2 non-editable install + canonical E2E | PASS |
| High | Unbounded latest dependency resolution에서 deterministic gate regression | NumPy 2.4/pandas 3.0/scikit-learn 1.9까지 허용되어 검증되지 않은 조합이 선택됨 | 검증된 minor-line upper bounds와 exact Python 3.13 qualification constraints 추가; metric/split/threshold는 변경하지 않음 | Q2/Q3 clean installs 및 동일 WAPE/cost/config hash 확인 | PASS |
| Medium | Module CLI와 console-script --version 문자열 불일치 | argparse version label이 entrypoint마다 달랐음 | support-capacity 1.4.1rc2로 통일 | CLI tests + Q2/Q3 outside-repo version checks | PASS |
| Medium | API request unknown fields가 silently ignored | Pydantic request models에 extra=forbid 미설정 | 모든 public request model을 strict schema로 전환 | staffing/pipeline unknown-field 422 tests + real-socket failure injection | PASS |
| Medium | Benchmark가 qualification-owned SIGTERM(-15)을 service failure로 오분류 | POSIX signal return-code semantics 미반영 | expected termination signal을 명시적으로 허용하고 다른 nonzero는 실패 | test_benchmark_api.py + exact 200-request benchmark | PASS |
| Medium | Clean copy에 __pycache__가 포함되어 traceback path가 이전 workspace를 가리킴 | clean-room copy exclusion 불완전 | cache/bytecode 제외 후 모든 이전 evidence 무효화 및 clean reset | 새 Unicode/space path의 import/traceback/source path 점검 | PASS |
| Medium | release_candidate_handoff.json이 rc1 fingerprint를 가리킴 | qualification 수정 후 handoff 미재생성 | 최종 frozen source에서 handoff와 dependency snapshot/checksum 재생성 | handoff referenced-file checksum validation | PASS |


## 5. Local qualification matrix

| Gate | Command | Environment | Exit code | Result | Evidence |
|---|---|---:|---:|---|---|
| Format | `python -m ruff format --check src tests scripts` | Q1 | 0 | PASS | `reports/qualification/final_static_ruff_format.log` |
| Lint | `python -m ruff check src tests scripts` | Q1 | 0 | PASS | `reports/qualification/final_static_ruff_lint.log` |
| Compile | `python -m compileall -q src scripts tests` | Q1 | 0 | PASS | `reports/qualification/final_static_compileall.log` |
| CI static | `python scripts/verify_ci.py` | Q1 | 0 | PASS | `reports/qualification/final_static_ci.log` |
| Local secret scan | `python scripts/security_scan.py` | Q1 | 0 | PASS | 150 files, no findings |
| Full suite #1 | isolated all + coverage | Q1 | 0 | PASS | 107 passed; 90% coverage |
| Full suite #2 | isolated all + coverage | Q1 | 0 | PASS | 107 passed; 90% coverage |
| Source install | non-editable install + `pip check` | Q2 | 0 | PASS | clean venv; outside-repo import/CLI/doctor |
| Source E2E | run + output verify + bundle replay | Q2 | 0 | PASS | `PASS_WITH_RECOURSE` |
| Build | `python -m build` | Q2 | 0 | PASS | wheel + sdist |
| Distribution audit | `scripts/verify_distribution.py` | Q2 | 0 | PASS | version/content verified |
| Wheel install | rebuilt wheel only + `pip check` | Q3 | 0 | PASS | separate venv |
| Wheel E2E | run + output verify + bundle replay | Q3 | 0 | PASS | replay error 7.11e-15 |
| API lifecycle | real Uvicorn process ×3 | Q1 | 0 | PASS | health/request/shutdown/restart |
| API failure injection | malformed/missing/unknown/type/path/port | Q1 | 0 | PASS | expected 422/403/port failure |
| Concurrency | two writers, same output | Q1 | 0 | PASS | second writer exit 75; original valid |
| Artifact corruption | corrupt copies + original verification | Q1 | 0 | PASS | corrupt exit 1; original exit 0 |
| API benchmark | 200 requests, concurrency 8 | Q1 | 0 | PASS | 0 errors; latency observational |
| Vulnerability DB audit | `pip-audit` | Q1 | nonzero | NOT_AVAILABLE | external DNS/index unavailable; not claimed PASS |

## 6. GitHub Actions qualification matrix

| Workflow/job | OS | Runtime | Exact commit | Result | Evidence level |
|---|---|---|---|---|---|
| `ci / quality-*` | ubuntu-latest | 3.11, 3.12, 3.13 | unavailable | CONFIGURED; local canonical commands pass | Q0/Q2 |
| `ci / quality-windows-*` | windows-latest | 3.11 | unavailable | CONFIGURED ONLY | Q0 |
| `ci / torch-quality` | ubuntu-latest | 3.11 | unavailable | CONFIGURED ONLY; local optional Torch unit tests pass | Q0/Q1 |
| `ci / smoke` | ubuntu-latest | 3.11 | unavailable | CONFIGURED; local equivalent passes | Q0/Q2/Q3 |
| `ci / insufficient-workforce-gate` | ubuntu-latest | 3.11 | unavailable | CONFIGURED; local equivalent returns expected ITERATE | Q0/Q1 |
| `codeql / analyze-python` | ubuntu-latest | hosted | unavailable | CONFIGURED ONLY | Q0 |

No GitHub-hosted run was read or dispatched. Therefore “GitHub PASS” is not claimed.

## 7. Pipeline verification

| Stage | Actual implementation | Input → output | Result |
|---|---|---|---|
| Input/config | `config.py`, strict Pydantic models | YAML → typed config | PASS |
| Validation | data contracts/config validators | schema/time/category constraints → accepted/rejected inputs | PASS |
| Synthetic fixture | `data/synthetic.py` | seed/config → intervals, contacts, agents, events | PASS |
| Feature engineering | `data/features.py` | ordered history → leakage-controlled supervised matrix | PASS |
| Forecast candidates | `forecasting/*` | features → probabilistic forecasts | PASS |
| Reliability | RCWE/calibration/scenarios | forecasts → calibrated coherent trajectories | PASS |
| Strategic capacity | `optimization/capacity.py` | scenarios → capacity plan | PASS |
| Tactical scheduling | `optimization/scheduler.py` | coverage/skills/shifts → schedule | PASS |
| Queue digital twin | `queueing/simulator.py` | demand/schedule → wait/SLA/abandonment | PASS |
| Intraday recourse | `optimization/recourse.py` | frozen plan + realized offered load → actions/repaired schedule | PASS |
| Decision gate | `reliability/release_gate.py` | forecast/decision metrics → PASS_WITH_RECOURSE or ITERATE | PASS |
| Publication | `artifacts.py` | completed run → atomic indexed artifact tree | PASS |
| Reload/serving | bundle verifier + FastAPI | trusted artifact/request → verified result | PASS |
| Reporting/monitoring | `reporting.py`, `monitoring.py` | run evidence → reports/snapshots | PASS |

Canonical run: `PASS_WITH_RECOURSE`, selected `lightgbm+rcwe`, one-step WAPE `0.380564862184`, total cost `9492.69`. The intentionally insufficient-workforce fixture returned `ITERATE` and was not misrepresented as success.

## 8. Failure-injection results

- Invalid API JSON, missing field, unknown field, and invalid content type: 422.
- Path traversal: 403.
- Second process on occupied port: failed as expected.
- Corrupted output and corrupted bundle: verification exit 1; untouched original remained valid.
- Concurrent writer: second writer rejected with exit 75; first publication remained valid and lock was removed.
- Interrupted publish, stale pointer, malformed worker output, worker timeout, and bundle mismatch: covered by passing regression tests.
- External dependency failures: NOT_APPLICABLE for the canonical offline route; optional model dependencies fail with explicit installation guidance.
- API startup/shutdown/restart: three consecutive real-process passes.

## 9. Performance and stability

| Metric | Result |
|---|---:|
| Requests | 200 + 20 warm-up |
| Concurrency | 8 |
| Errors | 0 |
| Error rate | 0.0% |
| Startup | 3.830 s |
| Throughput | 472.278 req/s |
| p50 | 16.048 ms |
| p95 | 18.437 ms |
| p99 | 22.863 ms |
| Max | 24.046 ms |
| RSS before/after | 310100 / 318860 KiB |
| FD before/after | 13 / 13 |
| Shutdown | expected qualification-owned SIGTERM, accepted; no residual server process |

Latency and memory are observations because no release threshold had been defined in advance. The enforced performance gate was error rate = 0.

## 10. Test summary

- Collected/passed: **107/107** per final suite.
- Failed: 0.
- Skipped: 0.
- Xfailed/xpassed: 0/0.
- Isolated modules: 32.
- Coverage: **90%**, gate 85%.
- Consecutive final full-suite passes: **2**.
- Canonical smoke passes represented in final evidence: **4** (Q2 source, prior Q3, root exact, final rebuilt Q3 wheel).
- API startup/shutdown passes: **3 consecutive**.

## 11. Build and artifact summary

| Artifact | Size | SHA-256 | Clean install/reload |
|---|---:|---|---|
| `support_capacity_reliability-1.4.1rc2-py3-none-any.whl` | 95216 bytes | `6b7c10e79149db4218d28b5bf988bc37f12dfe74cbed514120be6fe386851c8f` | Q3 install, pip check, E2E, artifact reload PASS |
| `support_capacity_reliability-1.4.1rc2.tar.gz` | 81718 bytes | `c0b7b88b7ab469667bfba74f417e3de79fa3ce466c90657d843ecad1938de646` | Build/content verification PASS |
| `release_candidate_handoff.json` | 17862 bytes | `f24809837d15e8e87b143ccc2d697e3f531d4a343567dc2a920046b47850d5a1` | referenced checksums validated |

Build command: `python -m build`. Wheel contained 46 entries and sdist 64 entries; no cache, report, output, `.env`, or Git metadata leakage was found.

## 12. Exact user commands

### Windows setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .[dev]
python -m pip check
python scripts\qualify_local.py --profile CORE --skip-install
```

### Linux setup and exact tested Python 3.13 dependencies

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --constraint constraints/qualification-py313.txt '.[dev]'
python -m pip check
```

### CPU base install

```bash
python -m pip install .
support-capacity doctor --config configs/smoke.yaml
```

### Optional Torch/GPU branch

```bash
# Choose the CPU/CUDA-specific torch command appropriate to the host first.
python -m pip install '.[torch]'
support-capacity validate-config --config configs/smoke_torch.yaml
```

### Test/full qualification

```bash
python scripts/qualify_local.py --profile EXTENDED --skip-install
```

### Sample run and artifact reload

```bash
support-capacity run --config configs/smoke.yaml --require-release
support-capacity verify-output --output outputs/smoke
support-capacity verify-model-bundle --artifact-dir outputs/smoke/artifacts
```

### Negative gate

```bash
support-capacity run --config configs/stress_insufficient_workforce.yaml --expected-status ITERATE
```

### API/service

```bash
uvicorn support_capacity_reliability.api.app:app --host 127.0.0.1 --port 8000
```

### Docker — configured, not executed here

```bash
docker build -t support-capacity-reliability:1.4.1rc2 .
docker run --rm -v "$PWD/outputs:/app/outputs" support-capacity-reliability:1.4.1rc2
```

### Cleanup

```bash
rm -rf .venv build dist .pytest_cache .ruff_cache .coverage* htmlcov
find . -type d -name __pycache__ -prune -exec rm -rf {} +
```

## 13. Known limitations

| Limitation | Impact | Condition | Workaround/follow-up | Release blocker? |
|---|---|---|---|---|
| No Q4 GitHub-hosted run | Cannot claim GitHub PASS | no remote run access | push exact source fingerprint and require all jobs | Yes for unconditional qualification |
| No Windows execution | Windows parity unproven | Linux-only environment | run canonical PowerShell qualification | Conditional item |
| No Docker executable | container runtime unproven | Docker unavailable | build/run non-root image and smoke | Conditional item |
| No GPU/Chronos execution | no CUDA/foundation-model claim | optional extras unavailable/unrequested | run full GPU doctor/E2E | Conditional item |
| `pip-audit` unavailable | no vulnerability DB conclusion | DNS/index unavailable | rerun with network access | Conditional item |
| Python 3.13 exact constraints only | not a universal lock | other OS/runtime | generate platform/runtime lock sets | No for tested scope |
| Synthetic/offline data | no live business claim | portfolio/replay setting | validate real governed data separately | No |
| Trusted joblib boundary | unsafe untrusted deserialization | external artifact | require checksum/provenance/trusted store | No when contract followed |

## 14. Files changed

Added 24 source files, including CodeQL/Dependabot configuration, exact qualification constraints, operations/troubleshooting docs, portable qualification wrappers, security/benchmark/evidence scripts, optional-Torch config, process utilities, and regression tests. Modified 25 source files, including CI, packaging metadata, README, canonical configs, CLI, API, runtime, artifact publication, factory/pipeline imports, and affected tests. No production source file was deleted; stale generated outputs/reports from the prior archive were replaced by current qualification evidence.

The exact source-level list is recorded in `reports/candidate/source_diff.json` and the full qualified manifest in `reports/candidate/source_manifest.json`.

## 15. Evidence ledger

The machine-readable chronological command ledger is `reports/candidate/evidence_log.json`. Primary final evidence includes:

1. Static integrity, lint, format, compile, CI, and local secret scans — PASS.
2. Final full suite #1 — 107/107, 90%, exit 0.
3. Final full suite #2 — 107/107, 90%, exit 0.
4. Q2 non-editable source install and `pip check` — PASS.
5. Q2 canonical E2E and artifact reload — PASS.
6. Final wheel/sdist build and distribution audit — PASS.
7. Q3 rebuilt-wheel-only install, `pip check`, outside-repo CLI/doctor, canonical E2E and reload — PASS.
8. Root canonical smoke and expected-negative stress — PASS/expected ITERATE.
9. API lifecycle ×3, strict invalid requests, port conflict, concurrency, and corruption tests — PASS.
10. API benchmark — 200/200 success, 0 errors.
11. Vulnerability DB query — NOT_AVAILABLE, explicitly not reported as PASS.
12. GitHub-hosted, Windows, Docker, and GPU execution — NOT_EXECUTED.

## 16. Final claim

검증한 **Linux x86_64 / Python 3.13.5 / CPU canonical path / clean non-editable source install / rebuilt-wheel-only install / CLI·API·E2E·artifact reload·failure-injection·concurrency·performance 관찰 범위** 내에서는 알려진 release blocker가 발견되지 않았으며, 기록된 gate가 통과했다.


## Post-upload lifecycle patch addendum

- Created at UTC: 2026-06-18T18:20:03.526883+00:00
- Updated source fingerprint: `c9215ea037fe46afc01683ab72a5cc687e67a391fcd3a66b6d2ad06b473e161e`
- Fix: console-script entrypoint now uses `support_capacity_reliability.cli:entrypoint` to flush stdio and hard-exit after CLI completion, reducing native-library shutdown/descriptor hang risk.
- Targeted verification: CLI/process/runtime/API/artifact/pipeline targeted tests passed; static gates, build, distribution verification, config validation, and doctor passed.
- Final verdict remains CONDITIONALLY QUALIFIED because GitHub-hosted, Windows, Docker, and GPU/Chronos paths were not executed.


## Final error hardening loop addendum

Source fingerprint after loop: `8f8f755f90eb73aef1195d3bf14663c21dcaf7897b0fa4bba4e5dd894e040e62`.

- Removed root `.coverage*` shards from the release bundle.
- Patched the public console `run` path to emit long-run JSON using `os.write` before hard process exit.
- Rebuilt wheel/sdist and reran targeted CLI/runtime/API/artifact tests plus static gates.
- Verdict remains CONDITIONALLY QUALIFIED because Q4 GitHub, Windows, Docker, GPU/Chronos, and full repeated suite after this final patch were not executed.
