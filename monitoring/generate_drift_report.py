from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "heart.csv"
OUTPUT_PATH = ROOT / "drift_report.html"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    reference, current = train_test_split(df, test_size=0.3, random_state=7)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    snapshot.save_html(str(OUTPUT_PATH))

    print(f"Reporte de drift guardado en: {OUTPUT_PATH}")
    print(f"Reference: {len(reference)} filas | Current: {len(current)} filas")


if __name__ == "__main__":
    main()
