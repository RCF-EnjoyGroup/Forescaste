"""Features de demanda histórica (spec ``feature-engineering``, 7.2).

Lags 1/7/14/28/365, rolling 7/14/28/56 (mean), mediana y desviación,
mismo día de la semana histórico y mismo mes histórico. Todo calculado con
:func:`src.features.asof.rolling_backward` / ``lag`` ⇒ exclusivamente hacia
atrás (orden temporal respetado) y **excluyendo la propia fila** (el target
nunca entra a sus propias features).

Semántica calendario (remediación): los lags y rollings se calculan sobre
la serie reindexada a frecuencia diaria por propiedad, de modo que ``lag_7``
siempre signifique "mismo día la semana pasada" y ``lag_365`` el mismo día
del año anterior, aunque la fuente tenga fechas faltantes (huecos). Sin esta
reindexación, un hueco de 93+ fechas desplaza silenciosamente el lag 365 a
otra estación del año.

Lag 365 solo se calcula cuando la propiedad cubre ≥ 1 año calendario; si
no, se omite para ESA propiedad y se registra en ``missing_features`` con
atribución por propiedad: ``{feature: [propiedades afectadas]}``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.asof import rolling_backward

LAGS = (1, 7, 14, 28, 365)
ROLLING_WINDOWS = (7, 14, 28, 56)
LAG_365_MIN_DAYS = 365


def add_demand_features(
    df: pd.DataFrame,
    date_col: str = "stay_date",
    value_col: str = "room_nights",
    group_col: str | None = "property",
    lags: tuple[int, ...] = LAGS,
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Construye features retrospectivas de demanda por propiedad.

    Returns
    -------
    (frame, missing_features):
        ``missing_features`` mapea cada feature omitida a las propiedades
        afectadas (p. ej. ``{"lag_365": ["SJOSL"]}``); una omisión local
        nunca se reporta como faltante global de la corrida.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.loc[out[date_col].notna()]
    sort_cols = [group_col, date_col] if group_col else [date_col]
    out = out.sort_values(sort_cols)
    missing: dict[str, list[str]] = {}

    groups = out.groupby(group_col, sort=False) if group_col else [(None, out)]

    def _daily_features(
        g: pd.DataFrame, label: str
    ) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
        """Features sobre la grilla diaria del grupo; se mapean de vuelta."""
        dates = pd.DatetimeIndex(g[date_col])
        if len(dates) == 0:
            return g, {}
        vals = pd.to_numeric(g[value_col], errors="coerce")
        # deduplicar fechas defensivamente (la calidad ya veta duplicados)
        daily = pd.Series(vals.to_numpy(dtype=float), index=dates).groupby(level=0).mean()
        daily = daily.sort_index()
        grid = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(grid)
        span_days = int(len(grid))

        feats: dict[str, pd.Series] = {}
        for k in lags:
            name = f"lag_{k}"
            if k >= LAG_365_MIN_DAYS and span_days < k:
                missing.setdefault(name, []).append(label)
                feats[name] = pd.Series(np.nan, index=grid)
                continue
            feats[name] = daily.shift(k)
        for w in rolling_windows:
            feats[f"rolling_mean_{w}"] = rolling_backward(daily, w, "mean")
        feats["rolling_median_28"] = rolling_backward(daily, 28, "median")
        feats["rolling_std_28"] = rolling_backward(daily, 28, "std")
        # mismo día de la semana histórico (calendario, no posicional)
        feats["same_dow_1w"] = daily.shift(7)
        feats["same_dow_2w"] = daily.shift(14)
        feats["same_dow_4w"] = daily.shift(28)
        # mismo mes histórico = mismo día del año anterior (calendario)
        if span_days >= LAG_365_MIN_DAYS:
            feats["same_month_year"] = daily.shift(365)
        else:
            missing.setdefault("same_month_year", []).append(label)
            feats["same_month_year"] = pd.Series(np.nan, index=grid)
        # media del mismo mes (histórico) — trailing 30 días, sin la fila propia
        feats["same_month_historical_mean"] = rolling_backward(daily, 30, "mean")
        return g, feats

    def _per_group(g: pd.DataFrame, label: str) -> pd.DataFrame:
        g, feats = _daily_features(g, label)
        if not feats:
            return g
        # mapear por VALOR de fecha (no por merge): preserva el índice original
        # de las filas — la suite de leakage alinea por índice y un merge lo
        # resetearía silenciosamente
        dates = pd.DatetimeIndex(pd.to_datetime(g[date_col], errors="coerce"))
        for name, s in feats.items():
            g[name] = s.reindex(dates).to_numpy(dtype=float)
        return g

    if group_col:
        pieces = [_per_group(g, str(key)) for key, g in groups]
        out = pd.concat(pieces).sort_values(sort_cols)
    else:
        out = _per_group(out, "__all__").sort_values(date_col)
    return out, {k: sorted(set(v)) for k, v in missing.items()}
