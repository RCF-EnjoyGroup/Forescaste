"""Pipeline de ingesta — la etapa previa a features/entrenamiento.

Orden (spec data-ingestion): leer → detectar esquema → validar → reportar →
gate. Si el gate lanza, nunca se llega a entrenar ni a escribir forecast.
Si se pasa la suite de leakage y detecta lookahead, el pipeline MUST fallar
(spec leakage-prevention).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import EngineConfig, load_config
from src.data.schema import SourceKind, SourceSchema, detect_schema
from src.validation.gate import enforce
from src.validation.leakage import run_leakage_suite
from src.validation.quality import validate_sources
from src.validation.report import DataQualityReport, build_report

FORECAST_FILENAME = "forecast.csv"


def load_sources(config: EngineConfig | None = None) -> dict[str, tuple[pd.DataFrame, SourceSchema]]:
    """Lee las tres fuentes desde las rutas de la config y detecta su esquema."""
    config = config or load_config()
    frames: dict[str, tuple[pd.DataFrame, SourceSchema]] = {}
    for name, path in config.source_paths.items():
        df = pd.read_csv(path)
        schema = detect_schema(df, SourceKind(name), config.column_hints)
        frames[name] = (schema.coerce(df), schema)
    return frames


def run_ingestion(
    config: EngineConfig | None = None,
    sources: dict[str, tuple[pd.DataFrame, SourceSchema]] | None = None,
    leakage: dict | None = None,
) -> DataQualityReport:
    """Ejecuta ingesta + validación + gate (+ suite de leakage si se pasa).

    Lanza ``CriticalDataError`` ante problemas críticos de calidad y
    ``LeakageError`` si la suite de leakage detecta lookahead.
    """
    config = config or load_config()
    if sources is None:
        sources = load_sources(config)
    issues = validate_sources(sources)
    report = build_report(sources, issues)
    enforce(report)  # detiene aquí si hay problemas críticos de calidad
    if leakage is not None:
        run_leakage_suite(**leakage)  # lanza LeakageError si hay lookahead
    return report


def forecast_output_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / FORECAST_FILENAME
