"""Smoke tests for detectors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pytest
from src.data_generator import generate_dataset
from src.detectors import StatisticalDetector, RuleBasedDetector, HybridDetector
from src.evaluator import evaluate

@pytest.fixture(scope="module")
def sample_df():
    return generate_dataset(n_samples=600, anomaly_ratio=0.08, seed=123)

def test_statistical(sample_df):
    r = StatisticalDetector().fit_predict(sample_df)
    assert len(r.labels) == len(sample_df)

def test_rules(sample_df):
    r = RuleBasedDetector().predict(sample_df)
    assert r.n_anomalies > 0

def test_hybrid(sample_df):
    r = HybridDetector().fit_predict(sample_df)
    m = evaluate(sample_df["is_anomaly"].values, r.labels, r.scores)
    assert 0 <= m["f1"] <= 1

def test_evaluator():
    m = evaluate(np.array([0,0,1,1,1]), np.array([0,1,1,1,0]))
    assert m["tp"] == 2 and m["fp"] == 1
