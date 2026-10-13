"""Validación de calidad de datos (spec ``data-ingestion``).

Cada check devuelve una lista de :class:`QualityIssue` con severidad
``error`` (crítico: detiene el pipeline) o ``warning`` (no detiene). Nunca
elimina registros: solo clasifica y reporta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from src.data.dates import parse_dates_flexible
from src.data.schema import SourceKind, SourceSchema

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class QualityIssue:
    """Un hallazgo de calidad: qué, dónde, cuántos registros y severidad."""

    check: str
    severity: Severity
    message: str
    records: int = 0
    properties: tuple[str, ...] = ()
    dates: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "records": self.records,
            "properties": list(self.properties),
            "dates": list(self.dates),
        }


def _issues(
    check: str,
    severity: Severity,
    message: str,
    mask: pd.Series,
    df: pd.DataFrame,
    schema: SourceSchema,
) -> list[QualityIssue]:
    if not bool(mask.any()):
        return []
    affected = df.loc[mask]
    props = tuple(
        sorted(str(p) for p in affected[schema.property_col].dropna().unique())
    )
    dates = tuple(
        sorted(str(pd.Timestamp(d).date()) for d in affected[schema.date_col].dropna().unique())
    )
    return [
        QualityIssue(
            check=check,
            severity=severity,
            message=f"{message}: {int(mask.sum())} registros",
            records=int(mask.sum()),
            properties=props,
            dates=dates,
        )
    ]


# ---------------------------------------------------------------- -----------


def _nullish(column: pd.Series) -> pd.Series:
    """NaN real o placeholder de nulo stringificado ('nan' tras astype(str)).

    ``_apply_aliases`` stringifica la columna de propiedad, por lo que una
    fila vacía del CSV llega con property='nan' y el resto NaN: ``isna``
    solo no la detecta.
    """
    if pd.api.types.is_datetime64_any_dtype(column):
        return column.isna()
    if pd.api.types.is_numeric_dtype(column):
        return column.isna()
    return column.isna() | (
        column.astype(str).str.strip().str.lower().isin({"nan", "none", ""})
    )


def check_empty_rows(df: pd.DataFrame) -> tuple[list[QualityIssue], pd.DataFrame]:
    """Filas completamente vacías (artefactos de exportación) → advertencia.

    Se excluyen de los demás checks: ``duplicated`` contaría NaN == NaN
    como duplicados (error espurio) y el análisis de huecos crashearía
    con grupos todo-NaT. Es higiene de datos, no corrupción: no detiene
    la corrida, pero queda reportado para Sistemas.
    """
    if df.empty:
        return [], df
    empty = pd.concat([_nullish(df[c]) for c in df.columns], axis=1).all(axis=1)
    empty.index = df.index
    if not bool(empty.any()):
        return [], df
    issue = QualityIssue(
        check="empty_rows", severity="warning",
        message=f"Filas completamente vacías: {int(empty.sum())} registros",
        records=int(empty.sum()),
    )
    return [issue], df.loc[~empty].copy()


def check_duplicates(df: pd.DataFrame, schema: SourceSchema) -> list[QualityIssue]:
    """(propiedad, fecha) duplicados → error.

    En pickup el grain es (propiedad, estancia, snapshot): ahí lo cubre
    :func:`check_duplicate_snapshots` para no duplicar hallazgos.
    """
    if schema.kind is SourceKind.PICKUP:
        return []
    dup = df.duplicated(subset=[schema.property_col, schema.date_col], keep=False)
    return _issues(
        "duplicate_property_date", "error",
        "Registros duplicados por propiedad/fecha", dup, df, schema,
    )


def check_negative_demand(df: pd.DataFrame, schema: SourceSchema) -> list[QualityIssue]:
    """Room Nights negativos → error (solo demanda/pickup)."""
    if schema.kind is SourceKind.CAPACITY:
        return []
    col = schema.roles.get("demand") or schema.roles.get("pickup")
    if col is None:
        return []
    mask = pd.to_numeric(df[col], errors="coerce") < 0
    return _issues("negative_rn", "error", "Room Nights negativos", mask, df, schema)


def check_negative_capacity(df: pd.DataFrame, schema: SourceSchema) -> list[QualityIssue]:
    if schema.kind is not SourceKind.CAPACITY:
        return []
    col = schema.col("capacity")
    mask = pd.to_numeric(df[col], errors="coerce") < 0
    return _issues("negative_capacity", "error", "Capacidad negativa", mask, df, schema)


def check_zero_capacity(df: pd.DataFrame, schema: SourceSchema) -> list[QualityIssue]:
    """Capacidad = 0 → advertencia (cierre/inventario sin operación)."""
    if schema.kind is not SourceKind.CAPACITY:
        return []
    col = schema.col("capacity")
    mask = pd.to_numeric(df[col], errors="coerce") == 0
    return _issues("zero_capacity", "warning", "Capacidad igual a cero", mask, df, schema)


def check_missing_dates(df: pd.DataFrame, schema: SourceSchema) -> list[QualityIssue]:
    """Huecos en la serie diaria por propiedad → advertencia.

    Las fechas se parsean con el parser flexible (ISO/DD-MM, spec
    data-ingestion); las presentes pero ilegibles para todo formato se
    reportan como error por grupo — sin crash.
    """
    issues: list[QualityIssue] = []
    dates = parse_dates_flexible(df[schema.date_col])
    if dates.isna().all():
        return [
            QualityIssue(
                check="missing_dates", severity="error",
                message="Columna de fecha ilegible (0 fechas parseadas)",
                records=len(df),
            )
        ]
    for prop, grp in df.assign(_date=dates).groupby(schema.property_col, dropna=False):
        if grp["_date"].isna().all():
            issues.append(
                QualityIssue(
                    check="missing_dates", severity="error",
                    message=(
                        f"Propiedad '{prop}': fechas presentes pero ilegibles "
                        f"en {len(grp)} registros"
                    ),
                    records=len(grp),
                    properties=(str(prop),),
                )
            )
            continue
        full = pd.date_range(grp["_date"].min(), grp["_date"].max(), freq="D")
        gaps = full.difference(pd.DatetimeIndex(grp["_date"].dropna().unique()))
        if len(gaps):
            issues.append(
                QualityIssue(
                    check="missing_dates",
                    severity="warning",
                    message=f"Propiedad '{prop}': {len(gaps)} fechas faltantes en la serie diaria",
                    records=len(gaps),
                    properties=(str(prop),),
                    dates=tuple(str(pd.Timestamp(d).date()) for d in sorted(gaps)),
                )
            )
    return issues


def check_capacity_change(df: pd.DataFrame, schema: SourceSchema) -> list[QualityIssue]:
    """Cambio de capacidad de más del 50% interdiario → advertencia."""
    if schema.kind is not SourceKind.CAPACITY:
        return []
    col = schema.col("capacity")
    work = df[[schema.property_col, schema.date_col, col]].copy()
    work[schema.date_col] = parse_dates_flexible(work[schema.date_col])
    work = work.sort_values([schema.property_col, schema.date_col])
    prev = work.groupby(schema.property_col)[col].shift(1)
    delta = (work[col] - prev).abs()
    ratio = pd.to_numeric(delta, errors="coerce") / pd.to_numeric(prev, errors="coerce")
    mask = ratio.fillna(0.0) > 0.5
    return _issues(
        "unexpected_capacity_change", "warning",
        "Cambio interdiario de capacidad > 50%", mask.fillna(False), work, schema,
    )


def check_invalid_lead(df: pd.DataFrame, schema: SourceSchema) -> list[QualityIssue]:
    """Lead no numérico o negativo → error (solo pickup)."""
    if schema.kind is not SourceKind.PICKUP:
        return []
    issues: list[QualityIssue] = []
    lead_col = schema.roles.get("lead")
    if lead_col and lead_col != "__derived__":
        raw = df[lead_col]
        non_numeric = raw.notna() & pd.to_numeric(raw, errors="coerce").isna()
        issues += _issues(
            "invalid_lead", "error", "Lead time no numérico", non_numeric.fillna(False), df, schema,
        )
        numeric = pd.to_numeric(raw, errors="coerce")
        issues += _issues(
            "negative_lead", "error", "Lead time negativo", (numeric < 0).fillna(False), df, schema,
        )
    elif lead_col == "__derived__":
        snap = schema.roles.get("snapshot")
        if snap and snap != "__derived__":
            stay = parse_dates_flexible(df[schema.date_col])
            snapshot = parse_dates_flexible(df[snap])
            derived = (stay - snapshot).dt.days
            issues += _issues(
                "negative_lead", "error", "Lead time negativo (stay < snapshot)",
                (derived < 0).fillna(False), df, schema,
            )
    return issues


def check_duplicate_snapshots(df: pd.DataFrame, schema: SourceSchema) -> list[QualityIssue]:
    """(propiedad, fecha estancia, snapshot/lead) duplicado → error."""
    if schema.kind is not SourceKind.PICKUP:
        return []
    subset = [schema.property_col, schema.date_col]
    snap = schema.roles.get("snapshot")
    if snap and snap != "__derived__":
        subset.append(snap)
    else:
        lead = schema.roles.get("lead")
        if lead and lead != "__derived__":
            subset.append(lead)
    dup = df.duplicated(subset=subset, keep=False)
    return _issues("duplicate_snapshots", "error", "Snapshots de pickup duplicados", dup, df, schema)


def check_pickup_inconsistency(df: pd.DataFrame, schema: SourceSchema) -> list[QualityIssue]:
    """lead_days no coincide con stay_date - snapshot_date → error."""
    if schema.kind is not SourceKind.PICKUP:
        return []
    snap = schema.roles.get("snapshot")
    lead_col = schema.roles.get("lead")
    if not snap or snap == "__derived__" or not lead_col or lead_col == "__derived__":
        return []
    stay = parse_dates_flexible(df[schema.date_col])
    snapshot = parse_dates_flexible(df[snap])
    expected = (stay - snapshot).dt.days
    actual = pd.to_numeric(df[lead_col], errors="coerce")
    mismatch = expected.notna() & actual.notna() & (expected != actual)
    return _issues(
        "pickup_inconsistent", "error",
        "Lead declarado inconsistente con stay_date - snapshot_date",
        mismatch.fillna(False), df, schema,
    )


def check_rn_over_capacity(
    demand_df: pd.DataFrame,
    demand_schema: SourceSchema,
    capacity_df: pd.DataFrame,
    capacity_schema: SourceSchema,
    error_rate_threshold: float = 0.01,
) -> list[QualityIssue]:
    """RN > capacidad: error si supera el umbral de frecuencia, si no advertencia."""
    d_col = demand_schema.col("demand")
    c_col = capacity_schema.col("capacity")
    demand = demand_df[[demand_schema.property_col, demand_schema.date_col, d_col]].copy()
    capacity = capacity_df[[capacity_schema.property_col, capacity_schema.date_col, c_col]].copy()
    demand[demand_schema.date_col] = parse_dates_flexible(demand[demand_schema.date_col])
    capacity[capacity_schema.date_col] = parse_dates_flexible(
        capacity[capacity_schema.date_col]
    )
    merged = demand.merge(
        capacity,
        left_on=[demand_schema.property_col, demand_schema.date_col],
        right_on=[capacity_schema.property_col, capacity_schema.date_col],
        how="inner",
        suffixes=("_d", "_c"),
    )
    if merged.empty:
        return []
    over = pd.to_numeric(merged[d_col], errors="coerce") > pd.to_numeric(
        merged[c_col], errors="coerce"
    )
    n_over = int(over.fillna(False).sum())
    if n_over == 0:
        return []
    rate = n_over / len(merged)
    severity: Severity = "error" if rate > error_rate_threshold else "warning"
    affected = merged.loc[over.fillna(False)]
    return [
        QualityIssue(
            check="rn_over_capacity",
            severity=severity,
            message=(
                f"Room Nights por encima de la capacidad en {n_over}/{len(merged)} "
                f"registros ({rate:.1%}, umbral {error_rate_threshold:.1%})"
            ),
            records=n_over,
            properties=tuple(sorted(str(p) for p in affected[demand_schema.property_col].unique())),
            dates=tuple(
                sorted(str(pd.Timestamp(d).date())
                        for d in affected[demand_schema.date_col].dropna().unique())
            ),
        )
    ]


# ---------------------------------------------------------------------------


def validate_source(df: pd.DataFrame, schema: SourceSchema) -> list[QualityIssue]:
    """Corre todos los checks aplicables a una fuente individual.

    Las filas completamente vacías se reportan como advertencia de
    higiencia y se excluyen de los demás checks (ver
    :func:`check_empty_rows`).
    """
    issues: list[QualityIssue] = []
    if df.empty:
        return [
            QualityIssue(
                check="empty_source", severity="error",
                message=f"Fuente '{schema.kind.value}' sin registros", records=0,
            )
        ]
    empty_issues, df = check_empty_rows(df)
    issues += empty_issues
    if df.empty:
        return issues
    issues += check_duplicates(df, schema)
    issues += check_negative_demand(df, schema)
    issues += check_negative_capacity(df, schema)
    issues += check_zero_capacity(df, schema)
    issues += check_missing_dates(df, schema)
    issues += check_capacity_change(df, schema)
    issues += check_invalid_lead(df, schema)
    issues += check_duplicate_snapshots(df, schema)
    issues += check_pickup_inconsistency(df, schema)
    return issues


def validate_sources(
    sources: dict[str, tuple[pd.DataFrame, SourceSchema]],
    error_rate_threshold: float = 0.01,
) -> list[QualityIssue]:
    """Valida todas las fuentes incluyendo el cruce demanda vs capacidad."""
    issues: list[QualityIssue] = []
    for df, schema in sources.values():
        issues += validate_source(df, schema)
    if "demand" in sources and "capacity" in sources:
        d_df, d_schema = sources["demand"]
        c_df, c_schema = sources["capacity"]
        issues += check_rn_over_capacity(d_df, d_schema, c_df, c_schema, error_rate_threshold)
    return issues
