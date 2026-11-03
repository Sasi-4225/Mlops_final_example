import json
from pathlib import Path

metrics_path = Path("artifacts/metrics.json")
if not metrics_path.exists():
    raise SystemExit("Metrics missing. Run train_model.py first.")
metrics = json.loads(metrics_path.read_text())
for key in ("accuracy", "f1", "roc_auc"):
    value = metrics.get(key)
    if value is None or not 0 <= value <= 1:
        raise SystemExit(f"Quality gate failed: invalid {key}={value}")
if metrics["test_rows"] < 10:
    raise SystemExit("Quality gate failed: test split too small.")
print("Quality gate passed (sanity checks only):", metrics)
