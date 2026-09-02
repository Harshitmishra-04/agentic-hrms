import csv
import os
from datetime import datetime, timezone
from threading import Lock
from typing import Optional, Union

PREDICTION_LOG_DIR = os.path.join("data", "predictions")
PREDICTION_LOG_PATH = os.path.join(PREDICTION_LOG_DIR, "prediction_log.csv")
_FIELDS = ["timestamp", "employee_id", "model_version", "probability", "risk_level"]
_lock = Lock()


def log_prediction(
    employee_id: Optional[Union[int, str]],
    model_version: str,
    probability: float,
    risk_level: str,
) -> None:
    """Append one attrition prediction to data/predictions/prediction_log.csv."""
    os.makedirs(PREDICTION_LOG_DIR, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "employee_id": "" if employee_id is None else employee_id,
        "model_version": model_version,
        "probability": f"{float(probability):.6f}",
        "risk_level": risk_level,
    }
    with _lock:
        write_header = (not os.path.exists(PREDICTION_LOG_PATH)) or os.path.getsize(
            PREDICTION_LOG_PATH
        ) == 0
        with open(PREDICTION_LOG_PATH, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
