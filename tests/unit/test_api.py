from fastapi.testclient import TestClient

from support_capacity_reliability.api.app import app


def test_health_endpoint():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok", "version": "1.4.1rc2"}


def test_staffing_endpoint():
    client = TestClient(app)
    response = client.post(
        "/required-staffing",
        json={
            "contacts_per_interval": 12,
            "interval_minutes": 30,
            "average_handle_time_seconds": 420,
            "patience_mean_seconds": 240,
        },
    )
    assert response.status_code == 200
    assert response.json()["agents"] >= 1


def test_pipeline_endpoint_rejects_path_traversal():
    client = TestClient(app)
    response = client.post("/run-pipeline", json={"config_path": "../outside.yaml"})
    assert response.status_code == 403


def test_pipeline_endpoint_rejects_non_yaml():
    client = TestClient(app)
    response = client.post("/run-pipeline", json={"config_path": "configs/readme.txt"})
    assert response.status_code == 400


def test_pipeline_endpoint_hides_internal_exception_details(monkeypatch):
    def fail(_):
        raise RuntimeError("sensitive internal path /secret/token")

    import importlib

    app_module = importlib.import_module("support_capacity_reliability.api.app")
    monkeypatch.setattr(app_module, "run_pipeline_isolated", fail)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/run-pipeline", json={"config_path": "configs/smoke.yaml"})
    assert response.status_code == 500
    assert "sensitive" not in response.text
    assert "inspect server logs" in response.json()["detail"]


def test_pipeline_endpoint_rejects_output_path_outside_repository(monkeypatch):
    import importlib

    app_module = importlib.import_module("support_capacity_reliability.api.app")
    config = app_module.load_config("configs/smoke.yaml")
    unsafe_project = config.project.model_copy(update={"output_dir": "/tmp/escaped-output"})
    unsafe_config = config.model_copy(update={"project": unsafe_project})
    monkeypatch.setattr(app_module, "load_config", lambda _: unsafe_config)
    client = TestClient(app)
    response = client.post("/run-pipeline", json={"config_path": "configs/smoke.yaml"})
    assert response.status_code == 403
    assert "outputs directory" in response.json()["detail"]


def test_staffing_endpoint_rejects_unknown_field():
    client = TestClient(app)
    response = client.post(
        "/required-staffing",
        json={
            "contacts_per_interval": 12,
            "average_handle_time_seconds": 420,
            "patience_mean_seconds": 240,
            "unexpected": "value",
        },
    )
    assert response.status_code == 422


def test_pipeline_endpoint_rejects_unknown_field():
    client = TestClient(app)
    response = client.post(
        "/run-pipeline",
        json={"config_path": "configs/smoke.yaml", "unexpected": "value"},
    )
    assert response.status_code == 422
