# Network Anomaly Detector — Full Scaffold

Production-style project layout for network anomaly detection.

Includes CLI, Streamlit dashboard, four detectors, synthetic data generator, evaluation metrics, tests, and docs.

## Quick start

```bash
cd nad-scaffold
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python cli.py demo
streamlit run dashboard/app.py
pytest tests/ -q
```

## Structure

```
nad-scaffold/
├── cli.py
├── src/          # data_generator, features, detectors, evaluator
├── dashboard/    # Streamlit UI
├── data/ models/ examples/ docs/ tests/
└── GUIDE.md README.md requirements.txt
```

## Sibling projects

- [nad-poc](https://github.com/himanshuj003/nad-poc) — single-script POC
- [nad-cli](https://github.com/himanshuj003/nad-cli) — CLI only
- [nad-dashboard](https://github.com/himanshuj003/nad-dashboard) — Streamlit only
- Combined: [network-anomaly-detector](https://github.com/himanshuj003/network-anomaly-detector)

## License

MIT
