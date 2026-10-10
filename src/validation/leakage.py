"""Pruebas automáticas de leakage (spec ``leakage-prevention``, tasks 3.3/3.4).

Detección de: RN futuros, pickup futuro, capacidad futura, features calculadas
con el dataset completo, rolling con observaciones futuras y splits aleatorios.
Si detecta leakage, el pipeline MUST fallar (lanza :class:`LeakageError`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.features.asof import LeakageError, enforce_cutoff

# ---------------------------------------------------------------------------
# 3.3 — veto de splits aleatorios
# ---------------------------------------------------------------------------

_RANDOM_SPLIT_TOKENS = {
    "random", "random_split", "train_test_split", "shuffle", "kfold",
    "random_state", "aleatorio", "stratified",
}
_TEMPORAL_TOKENS = {"expanding", "rolling", "origin", "temporal", "time", "sequential"}


def validate_split_config(split: dict[str, Any] | None) -> None:
    """Valida que la estrategia de validación sea exclusivamente temporal.

    Lanza :class:`LeakageError` si la configuración mezcla observaciones
    futuras en entrenamiento respecto a la ventana de evaluación (split
    aleatorio / shuffle / k-fold aleatorio).
    """
    if not split:
        return
    strategy = str(split.get("strategy", "")).lower()
    if strategy in _RANDOM_SPLIT_TOKENS or any(t in strategy for t in _RANDOM_SPLIT_TOKENS):
        raise LeakageError(
            f"Estrategia de split no temporal detectada: '{strategy}'. "
            "El spec prohíbe el split aleatorio train/test; use expanding/rolling origin."
        )
    if "shuffle" in split and split["shuffle"]:
        raise LeakageError("split.shuffle=true equivale a split aleatorio; prohibido por el spec")
    random_state = split.get("random_state")
    if random_state is not None and not any(t in strategy for t in _TEMPORAL_TOKENS):
        raise LeakageError(
            "split con random_state sin estrategia temporal explícita: "
            "posible split aleatorio (prohibido)"
        )
    train_end = split.get("train_end")
    val_start = split.get("val_start")
    if train_end is not None and val_start is not None:
        # convención semiabierta: train = [.., train_end), val = [val_start, ..]
        # => val_start == train_end es contiguo (sin solape); el solape real es
        # val_start < train_end (como TimeWindow/slice_window)
        if pd.Timestamp(val_start) < pd.Timestamp(train_end):
            raise LeakageError(
                f"split mixto: val_start ({val_start}) es anterior a train_end ({train_end}); "
                "el entrenamiento incluiría información de la ventana de evaluación"
            )


def validate_temporal_order(
    train: pd.DataFrame, val: pd.DataFrame, date_col: str
) -> None:
    """Verifica que toda la validación sea estrictamente posterior al entrenamiento."""
    train_dates = pd.to_datetime(train[date_col], errors="coerce")
    val_dates = pd.to_datetime(val[date_col], errors="coerce")
    if train_dates.empty or val_dates.empty:
        raise LeakageError("train/vacío: no se puede validar el orden temporal")
    if val_dates.min() <= train_dates.max():
        raise LeakageError(
            f"split con solape temporal: train llega a {train_dates.max().date()} "
            f"y validation inicia en {val_dates.min().date()}"
        )


# ---------------------------------------------------------------------------
# Suites de detección
# ---------------------------------------------------------------------------


@dataclass
class LeakageFinding:
    check: str
    message: str
    records: int = 0

    def __str__(self) -> str:
        return f"[{self.check}] {self.message}"


@dataclass
class LeakageReport:
    findings: list[LeakageFinding] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings

    def raise_if_dirty(self) -> None:
        if self.findings:
            details = "; ".join(str(f) for f in self.findings)
            raise LeakageError(f"Leakage detectado ({len(self.findings)} hallazgos): {details}")


def check_future_values(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    value_cols: Iterable[str],
    obs_date_col: str,
    check: str,
) -> LeakageFinding | None:
    """Filas cuyo valor observado es posterior al corte (RN/pickup/capacidad futuros)."""
    dates = pd.to_datetime(df[obs_date_col], errors="coerce")
    future = df.loc[dates > pd.Timestamp(cutoff)]
    if future.empty:
        return None
    return LeakageFinding(
        check=check,
        message=(
            f"{len(future)} filas con observación posterior al corte "
            f"{pd.Timestamp(cutoff).date()} en columnas {list(value_cols)}"
        ),
        records=len(future),
    )


def check_feature_whole_dataset(
    train: pd.DataFrame,
    full: pd.DataFrame,
    feature_fn,
    feature_cols: Iterable[str] = (),
) -> LeakageFinding | None:
    """Detecta features calculadas con el dataset completo (train + futuro).

    Contrato: ``feature_fn`` es la función que produjo las features almacenadas.
    Se recomputa sobre ``train`` y sobre ``full`` (recortado a train):

    - honesto: ``stored ≈ feature_fn(train)`` (las features de una fila solo
      dependen de filas anteriores o iguales, por lo que recalcular con el
      frame completo que las produjo da lo mismo en el prefijo train).
    - leakage: ``stored ≈ feature_fn(full)[train]`` pero
      ``stored ≉ feature_fn(train)`` ⇒ la salida en filas train depende de
      filas fuera de train (el "dataset completo" con futuro).
    """
    if train.empty:
        return None
    recomputed_train = feature_fn(train)
    recomputed_full = feature_fn(full)
    recomputed_full = recomputed_full.loc[train.index]
    cols = list(feature_cols) or [
        c for c in recomputed_train.columns if pd.api.types.is_numeric_dtype(recomputed_train[c])
    ]
    offenders: list[str] = []
    for col in cols:
        if col not in train.columns or col not in recomputed_train.columns:
            continue
        stored = pd.to_numeric(train[col], errors="coerce").to_numpy(dtype=float)
        honest = pd.to_numeric(recomputed_train[col], errors="coerce").to_numpy(dtype=float)
        leaky = pd.to_numeric(recomputed_full[col], errors="coerce").to_numpy(dtype=float)
        stored_vs_honest = np.isclose(stored, honest, equal_nan=True)
        stored_vs_full = np.isclose(stored, leaky, equal_nan=True)
        if stored_vs_full.all() and not stored_vs_honest.all():
            offenders.append(col)
    if offenders:
        return LeakageFinding(
            check="feature_computed_on_whole_dataset",
            message=(
                f"features calculadas con el dataset completo (coinciden con full y difieren "
                f"del recálculo solo-train): {offenders}"
            ),
            records=len(offenders),
        )
    return None


def check_rolling_with_future(
    df: pd.DataFrame,
    date_col: str,
    raw_col: str,
    feature_col: str,
    window: int,
    cutoff: pd.Timestamp,
) -> LeakageFinding | None:
    """Feature rolling almacenada que no coincide con la ventana solo-hacia-atrás.

    Recalcula la ventana ``window`` sobre ``raw_col`` en dirección pasada
    (``center=False``, min_periods=1) y en dirección futura/centrada. Si el
    valor almacenado en ``feature_col`` coincide con la versión futura/centrada
    y NO con la pasada para fechas ≤ cutoff, la feature incluye observaciones
    posteriores (lookahead).
    """
    if df.empty or window <= 1:
        return None
    if raw_col not in df.columns or feature_col not in df.columns:
        return None
    work = df[[date_col, raw_col, feature_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.sort_values(date_col).dropna(subset=[raw_col])
    raw = pd.to_numeric(work[raw_col], errors="coerce")
    stored = pd.to_numeric(work[feature_col], errors="coerce")
    backward = raw.shift(1).rolling(window=window, min_periods=1, center=False).mean()
    forward = raw.rolling(window=window, min_periods=1, center=True).mean()
    mask_dates = work[date_col] <= pd.Timestamp(cutoff)
    ok_back = np.isclose(stored, backward, equal_nan=True)
    ok_fwd = np.isclose(stored, forward, equal_nan=True)
    bad = mask_dates & ~ok_back & ok_fwd
    if bad.any():
        return LeakageFinding(
            check="rolling_includes_future",
            message=(
                f"rolling de {window}: {int(bad.sum())} fechas cuyo valor almacenado coincide "
                "con la ventana centrada/futura y no con la retrospectiva"
            ),
            records=int(bad.sum()),
        )
    return None


def check_random_split(split: dict[str, Any] | None) -> LeakageFinding | None:
    try:
        validate_split_config(split)
    except LeakageError as exc:
        return LeakageFinding(check="random_split", message=str(exc))
    return None


_ROLLING_STAT = {"mean", "std", "median", "min", "max", "sum"}
_ROLLING_NAME = re.compile(r"^rolling_(mean|std|median|min|max|sum)_(\d+)$")


def check_rolling_target_inclusion(
    df: pd.DataFrame,
    target_col: str,
    feature_col: str,
    window: int,
    stat: str = "mean",
    date_col: str | None = None,
    group_col: str | None = None,
) -> LeakageFinding | None:
    """Rolling feature cuya ventana incluye el target de la propia fila.

    La versión honesta excluye la fila propia (``shift(1)`` antes de la
    ventana); si el valor almacenado coincide con la versión que INCLUYE el
    target de la fila (y difiere de la honesta donde ambas existen), la
    matriz de entrenamiento ML filtra el target → leakage.

    Las referencias se recomputan sobre la GRILLA DIARIA por grupo (misma
    semántica calendario del motor): con huecos en la fuente, una ventana
    puramente posicional sobre filas apunta a otro conjunto de días y
    produce falsos positivos.
    """
    if df.empty or stat not in _ROLLING_STAT or feature_col not in df.columns:
        return None
    if target_col not in df.columns:
        return None

    def _one(frame: pd.DataFrame) -> int:
        if date_col is not None:
            if date_col not in frame.columns:
                return 0
            dates = pd.DatetimeIndex(pd.to_datetime(frame[date_col], errors="coerce"))
            frame = frame.loc[dates.notna()]
            dates = pd.DatetimeIndex(pd.to_datetime(frame[date_col], errors="coerce"))
            if dates.empty:
                return 0
            raw = (
                pd.Series(
                    pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float),
                    index=dates,
                )
                .groupby(level=0)
                .mean()
                .sort_index()
            )
            grid = pd.date_range(raw.index.min(), raw.index.max(), freq="D")
            daily = raw.reindex(grid)
            incl = getattr(daily.rolling(window=window, min_periods=1), stat)()
            excl = getattr(daily.shift(1).rolling(window=window, min_periods=1), stat)()
            stored = pd.to_numeric(frame[feature_col], errors="coerce").to_numpy(dtype=float)
            incl_at = incl.reindex(dates).to_numpy(dtype=float)
            excl_at = excl.reindex(dates).to_numpy(dtype=float)
        else:
            raw = pd.to_numeric(frame[target_col], errors="coerce")
            stored = pd.to_numeric(frame[feature_col], errors="coerce").to_numpy(dtype=float)
            incl_at = getattr(raw.rolling(window=window, min_periods=1), stat)().to_numpy(dtype=float)
            excl_at = getattr(raw.shift(1).rolling(window=window, min_periods=1), stat)().to_numpy(dtype=float)
        incl_vs_excl = ~np.isclose(incl_at, excl_at, equal_nan=True)
        stored_vs_incl = np.isclose(stored, incl_at, equal_nan=True)
        stored_vs_excl = np.isclose(stored, excl_at, equal_nan=True)
        bad = incl_vs_excl & stored_vs_incl & ~stored_vs_excl
        return int(np.nansum(bad))

    n_bad = 0
    if group_col and group_col in df.columns:
        for _, g in df.groupby(group_col, sort=False):
            n_bad += _one(g)
    else:
        n_bad += _one(df)
    if n_bad == 0:
        return None
    return LeakageFinding(
        check="rolling_includes_target",
        message=(
            f"rolling_{stat}_{window}: {n_bad} filas cuya feature incluye el target "
            "de la propia fila en la ventana (shift(1) ausente)"
        ),
        records=n_bad,
    )


def find_rolling_target_inclusions(
    df: pd.DataFrame,
    target_col: str,
    date_col: str | None = None,
    group_col: str | None = None,
) -> list[LeakageFinding]:
    """Escanea todas las columnas ``rolling_<stat>_<w>`` de un frame."""
    findings: list[LeakageFinding] = []
    for col in df.columns:
        m = _ROLLING_NAME.match(str(col))
        if not m:
            continue
        finding = check_rolling_target_inclusion(
            df, target_col, col, window=int(m.group(2)), stat=m.group(1),
            date_col=date_col, group_col=group_col,
        )
        if finding:
            findings.append(finding)
    return findings


def assert_no_target_in_rolling(
    df: pd.DataFrame,
    target_col: str,
    date_col: str | None = None,
    group_col: str | None = None,
) -> None:
    """Contrato duro: la matriz de features no contiene el target de la fila."""
    findings = find_rolling_target_inclusions(df, target_col, date_col, group_col)
    if findings:
        details = "; ".join(str(f) for f in findings)
        raise LeakageError(f"Leakage de target en features ({len(findings)}): {details}")


# ---------------------------------------------------------------------------
# Orquestador (3.4): inyectar leakage deliberado debe fallar la corrida
# ---------------------------------------------------------------------------


def run_leakage_suite(
    *,
    cutoff: pd.Timestamp,
    frames: dict[str, tuple[pd.DataFrame, str]],
    split: dict[str, Any] | None = None,
    train: pd.DataFrame | None = None,
    full: pd.DataFrame | None = None,
    feature_fn=None,
    feature_cols: Iterable[str] = (),
    rolling: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> LeakageReport:
    """Ejecuta todas las pruebas de leakage y devuelve el reporte.

    ``frames`` mapea nombre -> (df, columna de fecha de observación) para
    demand (RN), pickup y capacity; cualquier fila posterior al corte es
    leakage. ``rolling`` acepta un dict o una lista de dicts (un check por
    feature/serie, p. ej. una por propiedad).
    """
    report = LeakageReport()
    cutoff = pd.Timestamp(cutoff)

    for name, (df, date_col) in frames.items():
        finding = check_future_values(
            df, cutoff, value_cols=[c for c in df.columns if c != date_col],
            obs_date_col=date_col, check=f"future_{name}",
        )
        if finding:
            report.findings.append(finding)

    if split:
        finding = check_random_split(split)
        if finding:
            report.findings.append(finding)

    if train is not None and full is not None:
        finding = check_feature_whole_dataset(train, full, feature_fn=feature_fn or (lambda d: d),
                                              feature_cols=feature_cols)
        if finding:
            report.findings.append(finding)

    if rolling:
        roll_items = rolling if isinstance(rolling, list) else [rolling]
        for item in roll_items:
            finding = check_rolling_with_future(
                item["df"],
                item.get("date_col", "date"),
                item["raw_col"],
                item["feature_col"],
                int(item["window"]),
                cutoff,
            )
            if finding:
                report.findings.append(finding)

    report.raise_if_dirty()
    return report


def assert_as_of(df: pd.DataFrame, date_col: str, cutoff: pd.Timestamp, label: str = "") -> None:
    """Wrapper explícito del contrato de corte (usado por tests y pipeline)."""
    enforce_cutoff(df, date_col, pd.Timestamp(cutoff), label)
