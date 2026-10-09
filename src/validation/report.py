"""Data Quality Report (spec ``data-ingestion``).

Consolida los hallazgos de :mod:`src.validation.quality` en un reporte con
status, número de registros, errores, advertencias, propiedades afectadas y
fechas afectadas. El status determina si el pipeline puede continuar:

- ``fail``    → hay errores críticos; el pipeline MUST detenerse.
- ``warnings``→ solo advertencias; el pipeline continúa.
- ``pass``    → sin hallazgos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.validation.quality import QualityIssue


@dataclass
class DataQualityReport:
    """Reporte de calidad consolidado por corrida."""

    record_counts: dict[str, int] = field(default_factory=dict)
    issues: list[QualityIssue] = field(default_factory=list)
    sources: tuple[str, ...] = ()

    # -- derivados ---------------------------------------------------------
    @property
    def errors(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def total_records(self) -> int:
        return sum(self.record_counts.values())

    @property
    def affected_properties(self) -> list[str]:
        props: set[str] = set()
        for issue in self.issues:
            props.update(issue.properties)
        return sorted(props)

    @property
    def affected_dates(self) -> list[str]:
        dates: set[str] = set()
        for issue in self.issues:
            dates.update(issue.dates)
        return sorted(dates)

    @property
    def status(self) -> str:
        if self.error_count:
            return "fail"
        if self.warning_count:
            return "warnings"
        return "pass"

    @property
    def is_critical(self) -> bool:
        """Problema crítico: el pipeline debe detenerse (spec data-ingestion)."""
        return self.status == "fail"

    def checks_summary(self) -> dict[str, dict[str, int]]:
        """Conteo de hallazgos por check y severidad."""
        summary: dict[str, dict[str, int]] = {}
        for issue in self.issues:
            bucket = summary.setdefault(issue.check, {"error": 0, "warning": 0})
            bucket[issue.severity] += 1
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "sources": list(self.sources),
            "records": {
                "total": self.total_records,
                "by_source": dict(self.record_counts),
            },
            "errors": self.error_count,
            "warnings": self.warning_count,
            "affected_properties": self.affected_properties,
            "affected_dates": self.affected_dates,
            "checks": self.checks_summary(),
            "issues": [i.to_dict() for i in self.issues],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=True)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    def summary_message(self) -> str:
        """Mensaje de un renglón con el problema identificado (spec: 'mensaje que identifica el problema')."""
        parts = [f"status={self.status}"]
        for issue in self.errors:
            parts.append(f"[{issue.severity}] {issue.check}: {issue.message}")
        return " | ".join(parts)


def build_report(
    sources: dict[str, tuple[Any, Any]],
    issues: list[QualityIssue],
) -> DataQualityReport:
    """Construye el reporte a partir de las fuentes cargadas y sus hallazgos.

    Parameters
    ----------
    sources:
        ``{nombre: (DataFrame, schema)}`` ya detectados.
    issues:
        Hallazgos devueltos por :func:`src.validation.quality.validate_sources`.
    """
    record_counts = {name: len(df) for name, (df, _schema) in sources.items()}
    return DataQualityReport(
        record_counts=record_counts,
        issues=list(issues),
        sources=tuple(sources.keys()),
    )
