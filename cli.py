#!/usr/bin/env python3
"""Network Anomaly Detector CLI"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import click, pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from src.data_generator import generate_dataset
from src.detectors import get_detector
from src.evaluator import evaluate, print_report
console = Console()

@click.group()
@click.version_option("1.0.0")
def cli():
    """Network Anomaly Detector"""
    pass

@cli.command()
@click.option("-n","--samples",default=5000)
@click.option("-r","--ratio",default=0.05)
@click.option("-o","--output",default="data/flows.csv")
@click.option("--seed",default=42)
def generate(samples, ratio, output, seed):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(samples, ratio, seed, save_path=output)
    console.print(Panel.fit(f"[green]{len(df)} flows, {df['is_anomaly'].sum()} anomalies → {output}"))

@cli.command()
@click.option("-i","--input","input_path",required=True)
@click.option("-m","--method",default="hybrid",
    type=click.Choice(["statistical","isolation_forest","rule_based","hybrid"], case_sensitive=False))
@click.option("-o","--output",default=None)
@click.option("--threshold",default=3.5)
@click.option("--contamination",default=0.05)
@click.option("--min-votes",default=2)
def detect(input_path, method, output, threshold, contamination, min_votes):
    df = pd.read_csv(input_path)
    kw = {}
    if method=="statistical": kw["threshold"]=threshold
    elif method=="isolation_forest": kw["contamination"]=contamination
    elif method=="hybrid": kw.update({"contamination":contamination,"z_threshold":threshold,"min_votes":min_votes})
    result = get_detector(method, **kw).fit_predict(df)
    df["anomaly"]=result.labels; df["anomaly_score"]=result.scores
    console.print(f"Anomalies: {result.n_anomalies} / {len(df)}")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)

@cli.command(name="evaluate")
@click.option("-i","--input","input_path",required=True)
@click.option("-m","--method",default="hybrid",
    type=click.Choice(["statistical","isolation_forest","rule_based","hybrid"], case_sensitive=False))
def evaluate_cmd(input_path, method):
    df = pd.read_csv(input_path)
    result = get_detector(method).fit_predict(df)
    print_report(evaluate(df["is_anomaly"].values, result.labels, result.scores), method)

@cli.command()
@click.option("-n","--samples",default=2000)
def demo(samples):
    console.print(Panel.fit("[bold cyan]Demo[/bold cyan]"))
    df = generate_dataset(samples, 0.05, 42)
    for m in ["statistical","rule_based","hybrid"]:
        try:
            res = get_detector(m).fit_predict(df)
            met = evaluate(df["is_anomaly"].values, res.labels, res.scores)
            console.print(f"  {m:20s} found={res.n_anomalies:4d} F1={met['f1']:.3f}")
        except ImportError as e:
            console.print(f"  {m:20s} skipped ({e})")

if __name__ == "__main__":
    cli()
