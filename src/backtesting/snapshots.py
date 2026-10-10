"""Backtesting por snapshots históricos (spec ``backtesting``, tasks 7.1–7.3).

Para una fecha de estancia, simula el forecast cuando faltaban
90 / 60 / 30 / 14 / 7 / 3 / 1 día, con datasets por corte cacheados.

- 7.1 un snapshot histórico NO contiene datos posteriores a su corte.
- 7.2 las revisiones T-90 → T-1 → Actual son reconstruibles.
- 7.3 métricas globales y por banda de lead sobre los snapshots simulados.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from src.evaluation.breakdowns import metrics_by_lead_band
from src.evaluation.metrics import MetricResult, compute_metrics
from src.features.asof import AsOfBuilder, enforce_cutoff

DEFAULT_LEADS = (90, 60, 30, 14, 7, 3, 1)
SNAPSHOT_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}


@dataclass(frozen=True)
class Snapshot:
    """Un corte simulado para una fecha de estancia."""

    stay_date: pd.Timestamp
    forecast_date: pd.Timestamp   # stay_date - lead
    lead_days: int
    frame: pd.DataFrame           # solo datos observados <= forecast_date

    @property
    def label(self) -> str:
        return f"T-{self.lead_days}"


def build_snapshot(
    history: pd.DataFrame,
    stay_date: pd.Timestamp,
    lead_days: int,
    date_col: str = "stay_date",
    cache_key: str = "default",
) -> Snapshot:
    """7.1 — Recorta el histórico en ``stay_date - lead`` (con caché)."""
    stay_date = pd.Timestamp(stay_date)
    forecast_date = stay_date - pd.Timedelta(days=int(lead_days))
    key = (cache_key, str(stay_date.date()), str(forecast_date.date()))
    frame = SNAPSHOT_CACHE.get(key)
    if frame is None:
        builder = AsOfBuilder(date_col=date_col, cutoff=forecast_date)
        frame = builder.frame(history, f"snapshot T-{lead_days}")
        # invariante duro: nada posterior al corte
        enforce_cutoff(frame, date_col, forecast_date, f"snapshot T-{lead_days}")
        SNAPSHOT_CACHE[key] = frame
    return Snapshot(
        stay_date=stay_date,
        forecast_date=forecast_date,
        lead_days=int(lead_days),
        frame=frame,
    )


def clear_cache() -> None:
    SNAPSHOT_CACHE.clear()


@dataclass
class BacktestRevision:
    """Una revisión del forecast para una fecha de estancia."""

    stay_date: pd.Timestamp
    forecast_date: pd.Timestamp
    lead_days: int
    current_booked_rn: float
    expected_pickup_rn: float
    forecast_rn: float
    model_name: str = "backtest"

    def to_dict(self) -> dict:
        return {
            "stay_date": str(pd.Timestamp(self.stay_date).date()),
            "forecast_date": str(pd.Timestamp(self.forecast_date).date()),
            "lead_days": self.lead_days,
            "current_booked_rn": self.current_booked_rn,
            "expected_pickup_rn": self.expected_pickup_rn,
            "forecast_rn": self.forecast_rn,
            "model_name": self.model_name,
        }


@dataclass
class BacktestResult:
    """7.2 — Serie completa de revisiones + actuales observados."""

    revisions: list[BacktestRevision] = field(default_factory=list)
    actuals: dict[str, float] = field(default_factory=dict)

    def frame(self) -> pd.DataFrame:
        if not self.revisions:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.revisions])

    def series_for(self, stay_date: pd.Timestamp) -> pd.DataFrame:
        """T-90 → T-1 → Actual reconstruible para una fecha."""
        key = str(pd.Timestamp(stay_date).date())
        revs = [r.to_dict() for r in self.revisions if r.to_dict()["stay_date"] == key]
        out = pd.DataFrame(revs).sort_values("lead_days", ascending=False)
        actual = self.actuals.get(key)
        if actual is not None:
            out = pd.concat(
                [
                    out,
                    pd.DataFrame(
                        [
                            {
                                "stay_date": key,
                                "forecast_date": key,
                                "lead_days": 0,
                                "forecast_rn": actual,
                                "is_actual": True,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        return out

    def metrics(self) -> MetricResult:
        """7.3 — Métricas sobre todos los snapshots simulados."""
        if not self.revisions:
            raise ValueError("sin revisiones para calcular métricas")
        actuals, forecasts = [], []
        for r in self.revisions:
            a = self.actuals.get(str(pd.Timestamp(r.stay_date).date()))
            if a is None:
                continue
            actuals.append(a)
            forecasts.append(r.forecast_rn)
        if not actuals:
            raise ValueError("ninguna revisión tiene actual observado")
        return compute_metrics(actuals, forecasts)

    def metrics_by_lead(self) -> pd.DataFrame:
        """7.3 — Métricas por banda de lead sobre los snapshots."""
        if not self.revisions:
            return pd.DataFrame()
        rows = []
        for r in self.revisions:
            a = self.actuals.get(str(pd.Timestamp(r.stay_date).date()))
            if a is None:
                continue
            rows.append(
                {
                    "lead_days": r.lead_days,
                    "actual": a,
                    "forecast": r.forecast_rn,
                }
            )
        if not rows:
            return pd.DataFrame()
        return metrics_by_lead_band(pd.DataFrame(rows))


def run_backtest(
    history: pd.DataFrame,
    stay_dates: list[pd.Timestamp] | pd.DatetimeIndex,
    forecast_fn: Callable[[pd.DataFrame, pd.Timestamp, int], dict],
    leads: tuple[int, ...] = DEFAULT_LEADS,
    actuals: pd.Series | None = None,
    date_col: str = "stay_date",
    value_col: str = "room_nights",
    cache_key: str = "default",
) -> BacktestResult:
    """Ejecuta el backtest: para cada stay date × lead, congela y predice.

    Parameters
    ----------
    forecast_fn:
        ``(snapshot_frame, stay_date, lead) -> dict`` con al menos
        ``current_booked_rn``, ``expected_pickup_rn``, ``forecast_rn``.
    """
    result = BacktestResult()
    hist = history.copy()
    hist[date_col] = pd.to_datetime(hist[date_col])
    if actuals is not None:
        # Serie indexada por fecha (o Series alineada al history)
        if isinstance(actuals.index, pd.DatetimeIndex):
            result.actuals.update(
                {
                    str(k.date()): float(v)
                    for k, v in actuals.items()
                    if pd.notna(v)
                }
            )
        else:
            # indexado por posición del history
            for d, v in zip(hist[date_col], actuals):
                if pd.notna(v):
                    result.actuals[str(d.date())] = float(v)
    else:
        # por defecto, los reales observados viven en el mismo history
        for d, v in zip(hist[date_col], pd.to_numeric(hist[value_col], errors="coerce")):
            if pd.notna(v):
                result.actuals[str(d.date())] = float(v)

    for stay in stay_dates:
        for lead in leads:
            snap = build_snapshot(history, stay, lead, date_col, cache_key)
            # invariante 7.1 verificado dentro de build_snapshot
            out = forecast_fn(snap.frame, pd.Timestamp(stay), int(lead))
            result.revisions.append(
                BacktestRevision(
                    stay_date=pd.Timestamp(stay),
                    forecast_date=snap.forecast_date,
                    lead_days=int(lead),
                    current_booked_rn=float(out.get("current_booked_rn", float("nan"))),
                    expected_pickup_rn=float(out.get("expected_pickup_rn", float("nan"))),
                    forecast_rn=float(out["forecast_rn"]),
                    model_name=str(out.get("model_name", "backtest")),
                )
            )
    return result
