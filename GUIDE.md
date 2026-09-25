# User Guide

## Install
```bash
pip install -r requirements.txt
```

## CLI
```bash
python cli.py demo
python cli.py generate -n 5000 -o data/flows.csv
python cli.py detect -i data/flows.csv -m hybrid
python cli.py evaluate -i data/flows.csv -m hybrid
```

## Dashboard
```bash
streamlit run dashboard/app.py
```

## Tests
```bash
pytest tests/ -q
```

For defensive research only. Do not scan networks without permission.
