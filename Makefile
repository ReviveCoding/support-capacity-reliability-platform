PYTHON ?= python
CONFIG ?= configs/smoke.yaml
THREAD_ENV = OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=42 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

.PHONY: setup install doctor smoke stress full test coverage lint ci-static api api-smoke validate verify-output verify-model-bundle verify verify-smoke wheel-e2e regression-gate candidate-handoff clean package

setup:
	$(PYTHON) -m pip install .[dev]

install:
	$(PYTHON) -m pip install .

doctor:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) -m support_capacity_reliability.cli doctor --config $(CONFIG)

smoke:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) -m support_capacity_reliability.cli run --config configs/smoke.yaml --require-release
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) -m support_capacity_reliability.cli verify-output --output outputs/smoke
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) -m support_capacity_reliability.cli verify-model-bundle --artifact-dir outputs/smoke/artifacts

stress:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) -m support_capacity_reliability.cli run --config configs/stress_insufficient_workforce.yaml --expected-status ITERATE
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) -m support_capacity_reliability.cli verify-output --output outputs/stress_insufficient_workforce
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) -m support_capacity_reliability.cli verify-model-bundle --artifact-dir outputs/stress_insufficient_workforce/artifacts

full:
	$(MAKE) doctor CONFIG=configs/full.yaml
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) -m support_capacity_reliability.cli run --config configs/full.yaml

test:
	PYTHONPATH=src $(PYTHON) scripts/run_test_suite.py --suite all --summary-json reports/candidate/test_summary.json

coverage:
	PYTHONPATH=src $(PYTHON) scripts/run_test_suite.py --suite all --coverage --coverage-fail-under 85 --summary-json reports/candidate/test_summary.json

lint:
	$(PYTHON) -m ruff format --check src tests scripts
	$(PYTHON) -m ruff check src tests scripts

ci-static:
	$(PYTHON) scripts/verify_ci.py

api:
	$(THREAD_ENV) PYTHONPATH=src uvicorn support_capacity_reliability.api.app:app --host 0.0.0.0 --port 8000

api-smoke:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) scripts/smoke_api.py

verify-output:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) -m support_capacity_reliability.cli verify-output --output outputs/smoke

verify-model-bundle:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) -m support_capacity_reliability.cli verify-model-bundle --artifact-dir outputs/smoke/artifacts

validate:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) -m support_capacity_reliability.cli validate-config --config $(CONFIG)

verify: doctor lint ci-static coverage validate

verify-smoke: smoke

wheel-e2e: package
	PYTHONPATH=src $(PYTHON) scripts/smoke_installed_wheel.py --run-pipeline --run-api-pipeline

clean:
	rm -rf outputs/smoke outputs/stress_insufficient_workforce outputs/full \
		.pytest_cache .ruff_cache .coverage .coverage.* htmlcov build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

package:
	rm -rf build dist src/*.egg-info src/support_capacity_reliability.egg-info
	$(PYTHON) -m build
	PYTHONPATH=src $(PYTHON) scripts/verify_distribution.py
	PYTHONPATH=src $(PYTHON) scripts/smoke_installed_wheel.py


candidate-handoff: package
	PYTHONPATH=src $(PYTHON) scripts/create_release_handoff.py


regression-gate:
	PYTHONPATH=src $(PYTHON) scripts/verify_regression_gates.py
