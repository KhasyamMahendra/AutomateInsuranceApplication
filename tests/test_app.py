"""
Unit tests for the Insurance Premium Predictor Flask app.
Run with: pytest tests/ --cov=. --cov-report=term-missing -v
"""
import os
import sys
import subprocess
import pytest

# Ensure the project root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "insurance_model.pkl")


def _ensure_model():
    """Train the model if the pkl file does not exist yet."""
    if not os.path.exists(MODEL_PATH):
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "train_model.py")],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Model training failed:\n{result.stderr}")


# Train model once before the test session starts
_ensure_model()

from app import app as flask_app  # noqa: E402 – must come after model generation


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """Return a Flask test client with TESTING mode enabled."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


# ─────────────────────────────────────────────────────────────────────────────
# Route tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHomeRoute:
    def test_home_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_home_contains_form(self, client):
        response = client.get("/")
        assert b"<form" in response.data

    def test_home_no_prediction_initially(self, client):
        response = client.get("/")
        # The prediction block should not appear on a fresh GET
        assert b"Estimated Annual Cost" not in response.data


class TestPredictRoute:
    def test_predict_non_smoker_returns_200(self, client):
        response = client.post(
            "/predict",
            data={"age": "30", "bmi": "25.0", "smoker": "no"},
        )
        assert response.status_code == 200

    def test_predict_smoker_returns_200(self, client):
        response = client.post(
            "/predict",
            data={"age": "45", "bmi": "32.5", "smoker": "yes"},
        )
        assert response.status_code == 200

    def test_predict_result_contains_dollar_sign(self, client):
        response = client.post(
            "/predict",
            data={"age": "35", "bmi": "27.0", "smoker": "no"},
        )
        assert b"$" in response.data

    def test_predict_result_shown_in_page(self, client):
        response = client.post(
            "/predict",
            data={"age": "35", "bmi": "27.0", "smoker": "no"},
        )
        assert b"Estimated Annual Cost" in response.data

    def test_smoker_premium_higher_than_non_smoker(self, client):
        """Smoker premium should exceed non-smoker premium for the same profile."""
        def _extract_amount(data: bytes) -> float:
            # Find the dollar amount after the last '$' in the response
            decoded = data.decode("utf-8")
            idx = decoded.rfind("$")
            raw = decoded[idx + 1:].split("<")[0].replace(",", "").strip()
            return float(raw)

        non_smoker = client.post(
            "/predict",
            data={"age": "30", "bmi": "25.0", "smoker": "no"},
        )
        smoker = client.post(
            "/predict",
            data={"age": "30", "bmi": "25.0", "smoker": "yes"},
        )

        non_smoker_amount = _extract_amount(non_smoker.data)
        smoker_amount = _extract_amount(smoker.data)

        assert smoker_amount > non_smoker_amount, (
            f"Smoker ({smoker_amount}) should cost more than non-smoker ({non_smoker_amount})"
        )

    def test_age_affects_premium(self, client):
        """Older age should result in higher premium."""
        def _extract_amount(data: bytes) -> float:
            decoded = data.decode("utf-8")
            idx = decoded.rfind("$")
            raw = decoded[idx + 1:].split("<")[0].replace(",", "").strip()
            return float(raw)

        young = client.post("/predict", data={"age": "20", "bmi": "25.0", "smoker": "no"})
        old = client.post("/predict", data={"age": "60", "bmi": "25.0", "smoker": "no"})

        assert _extract_amount(old.data) > _extract_amount(young.data)

    def test_form_values_pre_filled_after_predict(self, client):
        """Submitted values should be echoed back into the form."""
        response = client.post(
            "/predict",
            data={"age": "42", "bmi": "28.3", "smoker": "yes"},
        )
        assert b"42" in response.data
        assert b"28.3" in response.data


# ─────────────────────────────────────────────────────────────────────────────
# Model artifact test
# ─────────────────────────────────────────────────────────────────────────────

class TestModelArtifact:
    def test_model_file_exists(self):
        assert os.path.exists(MODEL_PATH), "insurance_model.pkl must exist"

    def test_model_file_is_non_empty(self):
        assert os.path.getsize(MODEL_PATH) > 0, "insurance_model.pkl must not be empty"
