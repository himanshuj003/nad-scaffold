# Architecture

Data → Features → Detectors (Statistical / Isolation Forest / Rules / Hybrid) → CLI / Dashboard / Metrics

Modular detectors implement fit / predict / fit_predict and return DetectionResult.
