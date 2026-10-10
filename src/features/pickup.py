"""Features de pickup (spec ``feature-engineering``, 7.4).

Current booked RN, lead days, pickup de los últimos 1/3/7/14/30 días, stats
al mismo lead, volatilidad, aceleración/desaceleración, acumulado y % de la
demanda final. ``Pickup Acceleration = Recent Pickup - Previous Pickup``.

Todo se calcula con ventanas retrospectivas y el "pickup al mismo lead" usa
solo snapshots ≤ fecha de corte (as-of).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.asof import lag, rolling_backward

RECENT_WINDOWS = (1, 3, 7, 14, 30)


def add_pickup_features(
    df: pd.DataFrame,
    date_col: str = "stay_date",
    snapshot_col: str = "snapshot_date",
    pickup_col: str = "pickup_rn",
    lead_col: str = "lead_days",
    group_col: str | None = "property",
    recent_windows: tuple[int, ...] = RECENT_WINDOWS,
    cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Features de pickup por (propiedad, fecha de estancia) a lo largo del tiempo.

    Parameters
    ----------
    cutoff:
        Si se pasa, los snapshots posteriores al corte se descartan antes de
        calcular cualquier feature (contrato as-of).
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[snapshot_col] = pd.to_datetime(out[snapshot_col], errors="coerce")
    if cutoff is not None:
        out = out.loc[out[snapshot_col] <= pd.Timestamp(cutoff)].copy()

    sort = [group_col, date_col, snapshot_col] if group_col else [date_col, snapshot_col]
    out = out.sort_values(sort)

    def _per_key(g: pd.DataFrame) -> pd.DataFrame:
        res = g.copy()
        pk = res[pickup_col]
        # pickup reciente: variación del booked entre snapshots consecutivos
        delta = pk.diff()
        for w in recent_windows:
            res[f"pickup_last_{w}d"] = delta.rolling(window=w, min_periods=1).sum()
        # aceleración = pickup reciente - pickup del periodo anterior
        recent = delta.rolling(window=3, min_periods=1).sum()
        previous = delta.rolling(window=3, min_periods=1).sum().shift(3)
        res["pickup_acceleration"] = recent - previous
        res["pickup_deceleration"] = -res["pickup_acceleration"]
        # volatilidad
        res["pickup_volatility"] = delta.rolling(window=7, min_periods=2).std()
        # acumulado
        res["pickup_cumulative"] = pk.cumsum()
        return res

    if group_col and date_col:
        out = pd.concat(
            [_per_key(g) for _, g in out.groupby([group_col, date_col], sort=False)]
        )
    else:
        out = _per_key(out)

    # ---- pickup al mismo lead (as-of por construcción: ya filtró cutoff) ----
    if lead_col in out.columns:
        stats = (
            out.groupby([group_col, lead_col], sort=False)[pickup_col]
            .agg(
                pickup_hist_mean_same_lead="mean",
                pickup_hist_median_same_lead="median",
                pickup_hist_count_same_lead="count",
            )
            .reset_index()
        )
        merge_keys = [group_col, lead_col] if group_col else [lead_col]
        out = out.merge(stats, on=merge_keys, how="left")
    return out.sort_values(sort)


def pickup_pct_of_final(pickup_col: pd.Series, final_col: pd.Series) -> pd.Series:
    """Pickup como porcentaje de la demanda final (cero-seguro)."""
    return pd.to_numeric(pickup_col, errors="coerce") / pd.to_numeric(
        final_col, errors="coerce"
    ).replace(0, np.nan)
