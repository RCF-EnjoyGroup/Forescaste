"""Capa de negocio del forecast (spec ``forecasting``, tasks 5.5–5.8).

- 5.5 Restricción de capacidad: ``final = min(raw, capacity)`` conservando
  ``raw_forecast`` / ``capacity`` / ``final_forecast`` por separado + alerta.
- 5.6 Ocupación implícita 0–100% con detección de valores fuera de rango.
- 5.7 Jerarquía propiedad / portafolio con reconciliación y reporte de brecha.
- 5.8 Intervalos P10/P50/P90 (p10 <= p50 <= p90) y marcado de modelos sin
  intervalo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_OUTPUT_COLS = (
    "property",
    "stay_date",
    "raw_forecast_rn",
    "capacity",
    "final_forecast_rn",
)


@dataclass
class CapacityAlert:
    property: str
    stay_date: str
    raw_forecast: float
    capacity: float
    capped_by: float

    def to_dict(self) -> dict:
        return {
            "property": self.property,
            "stay_date": self.stay_date,
            "raw_forecast": self.raw_forecast,
            "capacity": self.capacity,
            "capped_by": self.capped_by,
        }


@dataclass
class BookedFloorAlert:
    """El campeón pronosticó por debajo de las reservas ya confirmadas."""

    property: str
    stay_date: str
    raw_forecast: float
    current_booked_rn: float
    shortfall: float  # booked - raw (cuánto faltaba llegar al piso)

    def to_dict(self) -> dict:
        return {
            "property": self.property,
            "stay_date": self.stay_date,
            "raw_forecast": self.raw_forecast,
            "current_booked_rn": self.current_booked_rn,
            "shortfall": self.shortfall,
        }


def apply_capacity_constraint(
    df: pd.DataFrame,
    capacity_col: str = "capacity",
    raw_col: str = "raw_forecast_rn",
    property_col: str = "property",
    date_col: str = "stay_date",
    booked_col: str | None = "current_booked_rn",
) -> tuple[pd.DataFrame, list]:
    """Piso de reservas + techo de capacidad (spec forecasting).

    ``final = min(max(raw, current_booked_rn), capacity)`` cuando existe
    on-the-books as-of el corte; sin booked → ``final = min(raw, capacity)``.
    El forecast publicado nunca queda por debajo de lo ya reservado (salvo
    que el propio booked exceda la capacidad: entonces manda la capacidad —
    físicamente inviolable — y se emite una alerta ``booked_over_capacity``).

    Returns
    -------
    (frame, alerts):
        ``frame`` conserva ``raw_forecast_rn`` intacto, agrega
        ``final_forecast_rn``, ``booked_floor_applied`` y
        ``capacity_constraint_applied``; ``alerts`` lista las violaciones
        detectadas ANTES de aplicar (capacidad → :class:`CapacityAlert`,
        piso/booked-sobre-capacidad → :class:`BookedFloorAlert`).
    """
    out = df.copy()
    if raw_col not in out.columns:
        raise KeyError(f"falta la columna de forecast bruto: '{raw_col}'")
    if capacity_col not in out.columns:
        raise KeyError(f"falta la columna de capacidad: '{capacity_col}'")

    raw = pd.to_numeric(out[raw_col], errors="coerce").to_numpy(dtype=float)
    cap = pd.to_numeric(out[capacity_col], errors="coerce").to_numpy(dtype=float)
    has_booked_col = booked_col is not None and booked_col in out.columns
    booked = (
        pd.to_numeric(out[booked_col], errors="coerce").to_numpy(dtype=float)
        if has_booked_col else np.full(len(out), np.nan)
    )

    alerts: list = []

    # piso de reservas: booked conocido y por encima del raw del campeón
    floor_binds = np.isfinite(booked) & np.isfinite(raw) & (raw < booked)
    floored = np.where(floor_binds, booked, raw)
    for i in np.flatnonzero(floor_binds):
        alerts.append(
            BookedFloorAlert(
                property=str(out.iloc[i][property_col]),
                stay_date=str(pd.Timestamp(out.iloc[i][date_col]).date()),
                raw_forecast=float(raw[i]),
                current_booked_rn=float(booked[i]),
                shortfall=float(booked[i] - raw[i]),
            )
        )
    # booked > capacidad: inconsistencia de DATOS (el piso no puede empujar
    # por encima del inventario físico) — se reporta, no se corrige aquí
    booked_over = np.isfinite(booked) & np.isfinite(cap) & (booked > cap)
    for i in np.flatnonzero(booked_over):
        alerts.append(
            BookedFloorAlert(
                property=str(out.iloc[i][property_col]),
                stay_date=str(pd.Timestamp(out.iloc[i][date_col]).date()),
                raw_forecast=float("nan"),
                current_booked_rn=float(booked[i]),
                shortfall=float(cap[i] - booked[i]),
            )
        )

    # alerta del TECHO antes del cap (sobre el valor ya pisado por el piso)
    violation = np.isfinite(floored) & np.isfinite(cap) & (floored > cap)
    for i in np.flatnonzero(violation):
        alerts.append(
            CapacityAlert(
                property=str(out.iloc[i][property_col]),
                stay_date=str(pd.Timestamp(out.iloc[i][date_col]).date()),
                raw_forecast=float(raw[i]),
                capacity=float(cap[i]),
                capped_by=float(floored[i] - cap[i]),
            )
        )

    out["final_forecast_rn"] = np.minimum(floored, cap)
    out["capacity_constraint_applied"] = violation
    out["booked_floor_applied"] = floor_binds
    return out, alerts


def compute_occupancy(
    df: pd.DataFrame,
    final_col: str = "final_forecast_rn",
    capacity_col: str = "capacity",
) -> pd.DataFrame:
    """5.6 — ``Forecast Occupancy = Forecast RN / Capacity`` (0–100%)."""
    out = df.copy()
    final = pd.to_numeric(out[final_col], errors="coerce")
    cap = pd.to_numeric(out[capacity_col], errors="coerce")
    out["forecast_occupancy"] = final / cap.replace(0, np.nan)
    out["occupancy_out_of_range"] = (
        (out["forecast_occupancy"] < 0) | (out["forecast_occupancy"] > 1)
    ).fillna(False)
    return out


def occupancy_violations(df: pd.DataFrame) -> pd.DataFrame:
    """5.6 — registros con ocupación fuera de 0–100%."""
    if "occupancy_out_of_range" not in df.columns:
        return df.iloc[0:0]
    return df.loc[df["occupancy_out_of_range"]]


@dataclass
class ReconciliationResult:
    portfolio: pd.DataFrame
    property_sum: pd.DataFrame
    gap_by_date: pd.DataFrame
    total_gap: float

    @property
    def gap_pct(self) -> float:
        total = float(self.property_sum["property_sum"].sum())
        return abs(self.total_gap) / total if total else 0.0


def reconcile_portfolio(
    df: pd.DataFrame,
    date_col: str = "stay_date",
    value_col: str = "final_forecast_rn",
    portfolio_forecast: pd.Series | None = None,
) -> ReconciliationResult:
    """5.7 — Suma por propiedad vs forecast de portafolio; reporta la brecha.

    Si no se provee un forecast de portafolio independiente, el portafolio
    reconciliado ES la suma de propiedades (bottom-up) y la brecha es 0; si se
    provee, se reporta la diferencia fecha a fecha.
    """
    prop_sum = (
        df.groupby(date_col, as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "property_sum"})
    )
    if portfolio_forecast is None:
        # bottom-up: el portafolio ES la suma de propiedades; brecha = 0
        portfolio = prop_sum.copy()
        portfolio["forecast_portfolio"] = portfolio["property_sum"]
        gap = portfolio[[date_col]].copy()
        gap["gap"] = 0.0
    else:
        port = pd.DataFrame(
            {date_col: pd.to_datetime(portfolio_forecast.index), "forecast_portfolio": portfolio_forecast.values}
        )
        port[date_col] = pd.to_datetime(port[date_col])
        prop_sum[date_col] = pd.to_datetime(prop_sum[date_col])
        portfolio = port.merge(prop_sum, on=date_col, how="outer")
        portfolio["forecast_portfolio"] = portfolio["forecast_portfolio"].fillna(0.0)
        portfolio["property_sum"] = portfolio["property_sum"].fillna(0.0)
        gap = portfolio[[date_col]].copy()
        gap["gap"] = portfolio["forecast_portfolio"] - portfolio["property_sum"]

    return ReconciliationResult(
        portfolio=portfolio,
        property_sum=prop_sum,
        gap_by_date=gap,
        total_gap=float(gap["gap"].sum()),
    )


@dataclass
class IntervalResult:
    """5.8 — Percentiles por registro + marcado de modelos sin intervalo."""

    frame: pd.DataFrame
    model_name: str
    has_intervals: bool
    note: str = ""

    def validate_ordering(self) -> None:
        """p10 <= p50 <= p90 en todos los registros (cuando existen)."""
        if not self.has_intervals:
            return
        f = self.frame
        bad = ~(
            (f["p10_forecast"] <= f["p50_forecast"] + 1e-9)
            & (f["p50_forecast"] <= f["p90_forecast"] + 1e-9)
        )
        if bad.any():
            raise ValueError(
                f"orden de percentiles violado en {int(bad.sum())} registros "
                f"(modelo {self.model_name})"
            )


def build_intervals(
    df: pd.DataFrame,
    model_name: str,
    point_col: str = "final_forecast_rn",
    residuals: pd.Series | None = None,
    levels: tuple[float, ...] = (0.10, 0.50, 0.90),
    residual_bands: dict[str, pd.Series] | None = None,
    lead_col: str = "lead_days",
    capacity_col: str = "capacity",
    min_residuals: int = 20,
    property_col: str | None = None,
) -> IntervalResult:
    """5.8 — Construye p10/p50/p90 anclados al forecast publicado.

    Remediación (spec forecasting):

    - El **p50 ES el punto publicado** (``point_col`` = forecast final
      restringido), no la predicción previa al cap.
    - ``p10``/``p90`` suman los cuantiles de residuos del backtest
      **de-mediaos** (spread puro alrededor del punto: el sesgo del campeón
      ya vive en el punto, no en la banda) y se recortan a
      ``[0, capacity]``: la banda jamás publica valores físicamente
      imposibles ni viola ``p10 <= p50 <= p90``.
    - Con ``residual_bands`` (residuos por banda de lead) la amplitud se
      calibra por banda; una banda con menos de ``min_residuals``
      observaciones hereda el pool global (documentado en ``note``).
    - Con ``residual_bands`` con claves ``(propiedad, banda)`` y
      ``property_col`` (residuos del forecast operativo), la amplitud se
      calibra por (propiedad, banda): una propiedad pequeña no hereda la
      escala de error del portafolio ni viceversa. Escalera de fallback:
      (propiedad, banda) ≥ ``min_residuals`` → pool de la propiedad →
      pool de la banda → global.
    - Si no hay residuos, marca el modelo como sin intervalo (spec:
      indicarlo explícito).
    """
    out = df.copy()
    anchor_note = ""
    if point_col not in out.columns and "raw_forecast_rn" in out.columns:
        # caller pre-restricción: anclar al punto disponible (documentado)
        anchor_note = f"anclado a '{point_col}' ausente; se usó 'raw_forecast_rn'"
        point_col = "raw_forecast_rn"
    if (residuals is None or len(residuals) == 0) and not residual_bands:
        out["p10_forecast"] = np.nan
        out["p50_forecast"] = np.nan
        out["p90_forecast"] = np.nan
        out["interval_available"] = False
        return IntervalResult(
            frame=out,
            model_name=model_name,
            has_intervals=False,
            note=f"modelo '{model_name}' sin intervalos disponibles",
        )

    point = pd.to_numeric(out[point_col], errors="coerce")
    cap = (
        pd.to_numeric(out[capacity_col], errors="coerce")
        if capacity_col in out.columns
        else None
    )

    def _shifts(res: pd.Series) -> tuple[float, float]:
        """Cuantiles de-mediaos: (q10 - q50, q90 - q50) — spread puro."""
        vals = pd.Series(res).dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            return 0.0, 0.0
        qs = np.quantile(vals, list(levels))
        return float(qs[0] - qs[1]), float(qs[2] - qs[1])

    global_shift = _shifts(residuals) if residuals is not None else None

    p10 = point.to_numpy(dtype=float).copy()
    p50 = point.to_numpy(dtype=float).copy()
    p90 = point.to_numpy(dtype=float).copy()
    notes: list[str] = []

    if residual_bands:
        from src.evaluation.breakdowns import lead_band

        prop_aware = (
            property_col is not None
            and property_col in out.columns
            and all(isinstance(k, tuple) for k in residual_bands)
        )
        if prop_aware:
            # pools de la escalera: (prop, banda) -> prop -> banda -> global
            prop_pool: dict[str, list[pd.Series]] = {}
            band_pool: dict[str, list[pd.Series]] = {}
            all_res: list[pd.Series] = []
            for (p, b), res in residual_bands.items():
                res = pd.Series(res).dropna()
                if len(res) == 0:
                    continue
                prop_pool.setdefault(str(p), []).append(res)
                band_pool.setdefault(str(b), []).append(res)
                all_res.append(res)
            prop_pool = {p: pd.concat(v, ignore_index=True) for p, v in prop_pool.items()}
            band_pool = {b: pd.concat(v, ignore_index=True) for b, v in band_pool.items()}
            global_series = pd.concat(all_res, ignore_index=True)

            row_bands = (
                out[lead_col].map(lead_band).to_numpy()
                if lead_col in out.columns
                else np.full(len(out), "__global__", dtype=object)
            )
            row_props = out[property_col].astype(str).to_numpy()

            shift_cache: dict[tuple[str, str], tuple[float, float]] = {}
            escalera: set[str] = set()
            for i, (p, b) in enumerate(zip(row_props, row_bands)):
                key = (str(p), str(b))
                if key not in shift_cache:
                    res = residual_bands.get((str(p), str(b)))
                    res = pd.Series(res).dropna() if res is not None else None
                    if res is not None and len(res) >= min_residuals:
                        shift_cache[key] = _shifts(res)
                    else:
                        lvl_prop = prop_pool.get(str(p))
                        lvl_band = band_pool.get(str(b))
                        if lvl_prop is not None and len(lvl_prop) >= min_residuals:
                            shift_cache[key] = _shifts(lvl_prop)
                            escalera.add(f"pool_propiedad:{p}")
                        elif lvl_band is not None and len(lvl_band) >= min_residuals:
                            shift_cache[key] = _shifts(lvl_band)
                            escalera.add(f"pool_banda:{b}")
                        else:
                            shift_cache[key] = _shifts(global_series)
                            escalera.add("pool_global")
                lo, hi = shift_cache[key]
                p10[i] = p10[i] + lo
                p90[i] = p90[i] + hi
            if escalera:
                notes.append(
                    f"celdas (propiedad, banda) con < {min_residuals} residuos heredan "
                    f"la escalera: {sorted(escalera)}"
                )
        else:
            bands = (
                out[lead_col].map(lead_band).to_numpy()
                if lead_col in out.columns
                else np.full(len(out), "__global__")
            )
            band_shifts: dict[str, tuple[float, float]] = {}
            fallback_bands: list[str] = []
            for band, res in residual_bands.items():
                res = pd.Series(res).dropna()
                if len(res) >= min_residuals:
                    band_shifts[band] = _shifts(res)
                else:
                    fallback_bands.append(band)
            if global_shift is None:
                pool = pd.concat(
                    [pd.Series(r).dropna() for r in residual_bands.values()], ignore_index=True
                )
                global_shift = _shifts(pool)
            if fallback_bands:
                notes.append(
                    f"bandas con < {min_residuals} residuos heredan el pool global: "
                    f"{sorted(set(fallback_bands))}"
                )
            for i, band in enumerate(bands):
                lo, hi = band_shifts.get(band, global_shift)
                p10[i] = p10[i] + lo
                p90[i] = p90[i] + hi
    elif global_shift is not None:
        lo, hi = global_shift
        p10 = p10 + lo
        p90 = p90 + hi

    if cap is not None:
        cap_vals = cap.to_numpy(dtype=float)
        p10 = np.minimum(np.maximum(p10, 0.0), cap_vals)
        p90 = np.minimum(np.maximum(p90, 0.0), cap_vals)
        capped = int(np.sum(p90 >= cap_vals - 1e-9))
        if capped:
            notes.append(f"p90 restringido a la capacidad en {capped} registros")
    else:
        p10 = np.maximum(p10, 0.0)
        p90 = np.maximum(p90, 0.0)

    out["p10_forecast"] = p10
    out["p50_forecast"] = p50
    out["p90_forecast"] = p90
    out["interval_available"] = True
    if anchor_note:
        notes.append(anchor_note)
    result = IntervalResult(
        frame=out,
        model_name=model_name,
        has_intervals=True,
        note="; ".join(notes),
    )
    result.validate_ordering()
    return result
