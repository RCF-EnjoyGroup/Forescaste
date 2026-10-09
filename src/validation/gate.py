"""Quality gate — detención del pipeline ante problemas críticos (spec ``data-ingestion``).

Si el Data Quality Report tiene status ``fail``, :func:`enforce` lanza
:class:`CriticalDataError` y el pipeline MUST detenerse antes de entrenar o
escribir forecast.
"""

from __future__ import annotations

from src.validation.report import DataQualityReport


class CriticalDataError(RuntimeError):
    """Problema crítico de calidad de datos: el pipeline se detiene."""

    def __init__(self, report: DataQualityReport):
        self.report = report
        super().__init__(
            "Problema crítico de calidad de datos — entrenamiento detenido. "
            f"{report.summary_message()} | "
            f"propiedades afectadas: {report.affected_properties or 'ninguna'} | "
            f"fechas afectadas: {len(report.affected_dates)}"
        )


def enforce(report: DataQualityReport) -> None:
    """Lanza :class:`CriticalDataError` si el reporte tiene errores críticos.

    Con solo advertencias no lanza: el pipeline continúa y las advertencias
    quedan registradas en el reporte (spec data-ingestion).
    """
    if report.is_critical:
        raise CriticalDataError(report)
