#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_generator import generate_dataset
from src.detectors import HybridDetector
from src.evaluator import evaluate, print_report

df = generate_dataset(1500, 0.06, 7)
r = HybridDetector().fit_predict(df)
print_report(evaluate(df["is_anomaly"].values, r.labels, r.scores), "hybrid")
