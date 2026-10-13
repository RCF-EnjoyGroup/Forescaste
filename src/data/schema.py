"""Detección automática de esquema para las tres fuentes (spec ``data-ingestion``).

Nunca asume nombres de columna: primero intenta los candidatos de la
configuración (``data.column_hints``) y, si no coinciden, aplica heurísticas de
dtype/cardinalidad. Si falta una columna mínima (propiedad, fecha, métrica; y
snapshot+lead para pickup) lanza :class:`SchemaDetectionError` para que el
pipeline se detenga antes de features/entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from src.config import load_config
from src.data.dates import parse_dates_flexible


class SourceKind(str, Enum):
    CAPACITY = "capacity"
    DEMAND = "demand"
    PICKUP = "pickup"


class SchemaDetectionError(ValueError):
    """No se pudo identificar el esquema mínimo de una fuente."""


# Columnas mínimas por fuente (spec data-ingestion: "estructura desconocida")
_REQUIRED_ROLES: dict[SourceKind, tuple[str, ...]] = {
    SourceKind.CAPACITY: ("property", "date", "capacity"),
    SourceKind.DEMAND: ("property", "date", "demand"),
    SourceKind.PICKUP: ("property", "date", "pickup", "snapshot", "lead"),
}


@dataclass(frozen=True)
class SourceSchema:
    """Mapeo rol -> columna real, detectado para una fuente concreta."""

    kind: SourceKind
    roles: dict[str, str]
    dtypes: dict[str, str]
    unmapped_columns: tuple[str, ...] = ()

    @property
    def property_col(self) -> str:
        return self.roles["property"]

    @property
    def date_col(self) -> str:
        return self.roles["date"]

    def col(self, role: str) -> str:
        try:
            return self.roles[role]
        except KeyError as exc:
            raise SchemaDetectionError(f"rol '{role}' no detectado en {self.kind.value}") from exc

    def coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza dtypes según el esquema: fechas a datetime, métricas a numérico.

        Si ``lead`` es derivable (stay_date - snapshot_date) la agrega como
        ``lead_days``. Columnas no mapeadas se conservan intactas. Las fechas
        se parsean con :func:`src.data.dates.parse_dates_flexible` (ISO y
        DD/MM, spec data-ingestion).
        """
        out = df.copy()
        out[self.roles["date"]] = parse_dates_flexible(out[self.roles["date"]])
        snap = self.roles.get("snapshot") if self.kind is SourceKind.PICKUP else None
        if snap and snap != "__derived__":
            out[snap] = parse_dates_flexible(out[snap])
        for role, col in self.roles.items():
            if role in ("property", "date", "snapshot") or col == "__derived__":
                continue
            out[col] = pd.to_numeric(out[col], errors="coerce")
        if self.roles.get("lead") == "__derived__":
            out["lead_days"] = (
                out[self.roles["date"]] - out[self.roles["snapshot"]]
            ).dt.days
        if self.roles.get("snapshot") == "__derived__":
            lead_col = self.roles.get("lead")
            if lead_col and lead_col != "__derived__":
                out["snapshot_date"] = out[self.roles["date"]] - pd.to_timedelta(
                    out[lead_col], unit="D"
                )
        return out


def _is_date_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype == object:
        sample = series.dropna().astype(str).head(20)
        if sample.empty:
            return False
        try:
            parsed = parse_dates_flexible(sample)
            return bool(parsed.notna().mean() >= 0.9)
        except (ValueError, TypeError):
            return False
    return False


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _pick_by_hint(roles: list[str], columns: list[str]) -> str | None:
    """Primera columna de ``columns`` cuyo nombre normalizado está en ``roles``."""
    norm = {_normalize(c): c for c in columns}
    for hint in roles:
        key = _normalize(hint)
        if key in norm:
            return norm[key]
    return None


def _normalize(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def detect_schema(
    df: pd.DataFrame,
    kind: SourceKind | str,
    hints: dict[str, list[str]] | None = None,
) -> SourceSchema:
    """Detecta propiedad/fecha/métrica (y snapshot/lead en pickup) en ``df``.

    Parameters
    ----------
    df:
        DataFrame crudo de la fuente.
    kind:
        Fuente: ``capacity`` | ``demand`` | ``pickup``.
    hints:
        Candidatos de nombre por rol (por defecto, los de ``config.yaml``).
    """
    kind = SourceKind(kind)
    # NOTA: un frame vacío pero con columnas identificables no falla aquí;
    # la vacuidad la reporta la validación de calidad (check empty_source).

    if hints is None:
        hints = load_config().column_hints

    columns = list(df.columns)
    roles: dict[str, str] = {}
    used: set[str] = set()

    # 1) Candidatos por nombre (config), por rol
    for role in _REQUIRED_ROLES[kind]:
        hint_key = role  # capacity/demand/pickup/date/property/snapshot/lead
        candidate = _pick_by_hint(hints.get(hint_key, []), [c for c in columns if c not in used])
        if candidate is not None:
            roles[role] = candidate
            used.add(candidate)

    # 2) Heurísticas para lo que falte
    if "date" not in roles:
        for col in columns:
            if col in used:
                continue
            if _is_date_like(df[col]):
                roles["date"] = col
                used.add(col)
                break

    metric_role = {"capacity": "capacity", "demand": "demand", "pickup": "pickup"}[kind.value]

    if "property" not in roles:
        for col in columns:
            if col in used or _is_date_like(df[col]) or _is_numeric(df[col]):
                continue
            # string/categorical de cardinalidad razonable = candidata a propiedad
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                nunique = df[col].nunique(dropna=True)
                if 0 < nunique <= max(50, len(df) // 10):
                    roles["property"] = col
                    used.add(col)
                    break

    if metric_role not in roles:
        for col in columns:
            if col in used or not _is_numeric(df[col]):
                continue
            roles[metric_role] = col
            used.add(col)
            break

    if kind is SourceKind.PICKUP:
        if "lead" not in roles:
            for col in columns:
                if col in used or not _is_numeric(df[col]):
                    continue
                if _normalize(col).rstrip("s") in ("lead", "lead_day", "lead_time", "day"):
                    roles["lead"] = col
                    used.add(col)
                    break
        if "lead" not in roles:
            # derivable: stay_date - snapshot_date si ambas existen
            if "snapshot" in roles and "date" in roles:
                roles["lead"] = "__derived__"
        if "snapshot" not in roles:
            for col in columns:
                if col in used:
                    continue
                if _normalize(col) in ("snapshot", "snap", "as_of", "bookings_date"):
                    roles["snapshot"] = col
                    used.add(col)
                    break
            if "snapshot" not in roles:
                for col in columns:
                    if col in used or not _is_date_like(df[col]) or col == roles.get("date"):
                        continue
                    roles["snapshot"] = col
                    used.add(col)
                    break
        if "snapshot" not in roles and "date" in roles:
            lead_col = roles.get("lead")
            if lead_col and lead_col != "__derived__":
                # sin columna snapshot: se deriva como stay_date - lead_days
                roles["snapshot"] = "__derived__"

    # 3) Validación de mínimos
    missing = [r for r in _REQUIRED_ROLES[kind] if r not in roles]
    if missing:
        raise SchemaDetectionError(
            f"fuente '{kind.value}': no se pudieron identificar las columnas mínimas "
            f"{missing}. Columnas disponibles: {columns}. "
            f"Decláralas en config.yaml (data.column_hints) si el nombre difiere."
        )

    dtypes = {role: str(df[col].dtype) for role, col in roles.items() if col != "__derived__"}
    unmapped = tuple(c for c in columns if c not in used)
    return SourceSchema(kind=kind, roles=roles, dtypes=dtypes, unmapped_columns=unmapped)
