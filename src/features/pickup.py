"""Features de pickup (spec ``feature-engineering``, 7.4).

Current booked RN, lead days, pickup de los últimos 1/3/7/14/30 días, stats
al mismo lead, volatilidad, aceleración/desaceleración, acumulado y % de la
demanda final. ``Pickup Acceleration = Recent Pickup - Previous Pickup``.

**Ventanas en días CALENDARIO** (fix fix-pickup-calendar-windows): la base
toma snapshots diarios desde 2024 y esparcidos antes (~5 por estancia); una
ventana de "últimos 7 días" implementada como "últimos 7 snapshots" contaba
hasta 7 semanas de booking en la era pre-2024. El nivel del on-the-books se
reindexa a grilla diaria por (propiedad, estancia) con forward-fill y las
ventanas operan sobre los incrementos diarios: con snapshots diarios es
equivalente a la versión por snapshots; con cadencia esparcida cuenta días
reales.

``pickup_cumulative`` acumula por SNAPSHOT (el spec no define su unidad;
ningún modelo la consume). ``pickup_pct_of_final`` divide por el target real:
solo para análisis, jamás como feature de entrenamiento.

Todo se calcula con ventanas retrospectivas y el "pickup al mismo lead" usa
solo snapshots ≤ fecha de corte (as-of).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RECENT_WINDOWS = (1, 3, 7, 14, 30)
ACCEL_WINDOW_DAYS = 7  # aceleración: últimos 7 días vs 7 días previos
VOLATILITY_WINDOW_DAYS = 7


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
    # la grilla diaria necesita fechas de snapshot legibles; NaT no puede
    # ubicarse en el calendario
    out = out.loc[out[snapshot_col].notna()].copy()

    sort = [group_col, date_col, snapshot_col] if group_col else [date_col, snapshot_col]
    out = out.sort_values(sort)

    def _per_key(g: pd.DataFrame) -> pd.DataFrame:
        """Ventanas de días calendario vía cumsums (rápido, sin rollings).

        El nivel del on-the-books se proyecta a posiciones de día calendario
        (``pos = snapshot − min``); el incremento diario es 0 en días sin
        snapshot y el salto exacto el día del snapshot. Toda ventana de w
        días terminando en ``p`` es ``cs[p+1] − cs[max(0, p−w+1)]`` sobre el
        cumsum de incrementos — equivalente exacto a la grilla diaria con
        ``rolling(w).sum()``, pero O(1) por fila.
        """
        res = g.copy()
        level = pd.Series(
            pd.to_numeric(res[pickup_col], errors="coerce").to_numpy(dtype=float),
            index=pd.DatetimeIndex(res[snapshot_col]),
        ).sort_index()
        level = level.loc[level.index.notna() & level.notna()]
        if level.empty:
            for w in recent_windows:
                res[f"pickup_last_{w}d"] = np.nan
            res["pickup_acceleration"] = np.nan
            res["pickup_deceleration"] = np.nan
            res["pickup_volatility"] = np.nan
            res["pickup_cumulative"] = res[pickup_col].cumsum()
            return res

        level = level[~level.index.duplicated(keep="last")]
        dmin = level.index.min()
        snap_pos = (level.index - dmin).days.to_numpy(dtype=int)
        vals = level.to_numpy(dtype=float)
        grid_len = int(snap_pos[-1]) + 1
        inc = np.zeros(grid_len, dtype=float)
        if len(snap_pos) > 1:
            inc[snap_pos[1:]] = np.diff(vals)
        cs = np.concatenate([[0.0], np.cumsum(inc)])
        cs2 = np.concatenate([[0.0], np.cumsum(inc * inc)])

        # posición de cada FILA del grupo (fechas duplicadas → misma posición).
        # Filas con pickup NaN quedan fuera del nivel: su posición puede
        # caer fuera del grid → features NaN (como reindex en la versión pandas)
        row_pos_raw = (pd.DatetimeIndex(res[snapshot_col]) - dmin).days.to_numpy(dtype=int)
        in_grid = (row_pos_raw >= 0) & (row_pos_raw < grid_len)
        row_pos = np.where(in_grid, row_pos_raw, 0)

        def _masked(values: np.ndarray) -> np.ndarray:
            return np.where(in_grid, values, np.nan)

        for w in recent_windows:
            lo = np.clip(row_pos - int(w) + 1, 0, None)
            res[f"pickup_last_{w}d"] = _masked(cs[row_pos + 1] - cs[lo])

        # aceleración = últimos 7 días − 7 días previos (NaN antes de 7 días)
        w_a = ACCEL_WINDOW_DAYS
        recent = cs[row_pos + 1] - cs[np.clip(row_pos - w_a + 1, 0, None)]
        prev_pos = row_pos - w_a
        accel = np.full(len(row_pos), np.nan)
        valid = (prev_pos >= 0) & in_grid
        if valid.any():
            pp = prev_pos[valid]
            previous = cs[pp + 1] - cs[np.clip(pp - w_a + 1, 0, None)]
            accel[valid] = recent[valid] - previous
        res["pickup_acceleration"] = accel
        res["pickup_deceleration"] = -accel

        # volatilidad = std (ddof=1) de los incrementos diarios en 7 días
        w_v = VOLATILITY_WINDOW_DAYS
        n = np.minimum(row_pos + 1, w_v)
        end = row_pos + 1
        start = end - n
        s = cs[end] - cs[start]
        s2 = cs2[end] - cs2[start]
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (s2 - (s * s) / n) / np.maximum(n - 1, 1)
        vol = np.where(n >= 2, np.sqrt(np.clip(var, 0.0, None)), np.nan)
        res["pickup_volatility"] = _masked(vol)

        # acumulado (por snapshot; el spec no define su unidad)
        res["pickup_cumulative"] = res[pickup_col].cumsum()
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
