"""
generate_drift_report.py
=========================
Etapa 6 del proyecto: monitoreo de data drift con Evidently.

Compara una partición "reference" (los datos con los que se entrenó el
modelo) contra una partición "current" (que simula un lote nuevo de datos
en producción) y genera un reporte HTML interactivo con las métricas de
data drift de Evidently.

Nota: como no hay un flujo de datos de producción real para este proyecto
de curso, "current" se simula con un split aleatorio separado del mismo
`heart.csv`. En un despliegue real, "current" vendría de las peticiones
que la API (Etapa 3) recibe día a día.

Uso:
    python monitoring/generate_drift_report.py
"""
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

    # reference = datos "de entrenamiento"; current = lote "nuevo" simulado
    reference, current = train_test_split(df, test_size=0.3, random_state=7)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    snapshot.save_html(str(OUTPUT_PATH))

    print(f"Reporte de drift guardado en: {OUTPUT_PATH}")
    print(f"Reference: {len(reference)} filas | Current: {len(current)} filas")


if __name__ == "__main__":
    main()
