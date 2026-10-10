"""Features de capacidad (spec ``feature-engineering``, 7.3).

Capacidad diaria, día anterior, cambio, rolling, utilización histórica y
capacidad restante. La capacidad es restricción operacional: estas features
informan al modelo pero el techo se aplica después (spec forecasting).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.asof import lag, rolling_backward


def add_capacity_features(
    df: pd.DataFrame,
    date_col: str = "stay_date",
    capacity_col: str = "capacity",
    booked_col: str | None = "current_booked_rn",
    group_col: str | None = "property",
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    sort = [group_col, date_col] if group_col else [date_col]
    out = out.sort_values(sort)

    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        res = g.copy()
        cap = res[capacity_col]
        res["capacity_prev_day"] = lag(cap, 1)
        res["capacity_change"] = cap - res["capacity_prev_day"]
        res["capacity_rolling_mean_7"] = rolling_backward(cap, 7, "mean")
        # utilización histórica: demanda observada / capacidad (solo pasado)
        if booked_col and booked_col in res.columns:
            res["capacity_remaining"] = cap - res[booked_col]
            res["capacity_utilization"] = res[booked_col] / cap.replace(0, np.nan)
        return res

    if group_col:
        out = pd.concat([_per_group(g) for _, g in out.groupby(group_col, sort=False)])
    else:
        out = _per_group(out)
    return out.sort_values(sort)
