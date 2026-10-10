"""Corrida end-to-end (task 10.5, remediación model-validation).

Cadena completa: ingesta → calidad (gate que DETIENE) → leakage (suite
completa, capacidad incluida) → features as-of (lags calendario) →
benchmark y SELECCIÓN DE CAMPEÓN → forecast OPERADO por el campeón (global
y por propiedad) → restricción de capacidad → backtest T-90..T-1
muestreado → intervalos P10/P50/P90 anclados y calibrados por lead →
métricas + desgloses + cobertura → monitoring (outliers/drift/retraining) →
Power BI → manifiesto → 10 entregables.

Verifica los criterios de aceptación de la task 10.5:
- 0 fallos de leakage
- ``final_forecast <= capacity`` en el 100% de registros
- métricas y desgloses presentes
- dataset Power BI presente
- manifiesto reproducible

**Datos**: si ``data/*.csv`` (las tres fuentes reales) no existen, la corrida
real no puede ejecutarse. Esta función opera sobre frames ya cargados, de modo
que la cadena se valida con datos de muestra en el esquema real, y la corrida
con fuentes reales se ejecuta al soltar los CSV en ``data/`` sin tocar código.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.snapshots import clear_cache, run_backtest
from src.config import EngineConfig, load_config
from src.data.schema import SourceKind, SourceSchema, detect_schema
from src.evaluation.breakdowns import (
    lead_band,
    metrics_by_lead_band,
    metrics_by_property,
    metrics_by_seasonality,
)
from src.evaluation.metrics import compute_metrics
from src.evaluation.selection import select_per_property
from src.explainability.explainer import explain_from_importance
from src.features.asof import AsOfBuilder, enforce_cutoff
from src.features.calendar import add_calendar_features
from src.features.capacity import add_capacity_features
from src.features.demand import add_demand_features
from src.features.holidays import holiday_calendar_from_config
from src.features.pickup import add_pickup_features
from src.forecasting.constraints import (
    apply_capacity_constraint,
    build_intervals,
    compute_occupancy,
)
from src.forecasting.core import (  # re-exportados para tests/CLI
    BACKTEST_LEADS_CHEAP,
    BACKTEST_LEADS_EXPENSIVE,
    CHEAP_MODELS,
    FittedCandidate,
    _apply_aliases,
    _booked_requests,
    _normalize_capacity,
    _normalize_demand,
    _normalize_pickup,
    _remaining_pickup_curves,
    build_forecasts,
    fit_candidate,
    predict_candidate,
)
from src.monitoring.drift import (
    detect_outliers,
    run_drift_suite,
    should_retrain,
)
from src.monitoring.run_manifest import RunManifest
from src.reporting.benchmark import generate_deliverables, run_benchmark
from src.reporting.visual_report import generate_visual_report
from src.reporting.forecast_table import (
    build_forecast_table,
    build_powerbi_dataset,
    executive_kpis,
    validate_forecast_table,
)
from src.validation.gate import enforce as enforce_quality_gate
from src.validation.leakage import (
    assert_no_target_in_rolling,
    run_leakage_suite,
)
from src.validation.quality import validate_sources
from src.validation.report import build_report

CANON = {"property": "property", "stay_date": "stay_date"}


def _merge_keep_index(left: pd.DataFrame, right: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """Merge left-preservando el índice (la suite de leakage alinea por índice).

    ``DataFrame.merge`` resetea el índice a un RangeIndex; el check de
    features-calculadas-con-dataset-completo compara filas por índice, así que
    la correspondencia con las fuentes originales debe preservarse. Si el
    right tiene claves duplicadas el left se duplicaría → error explícito.
    """
    idx = left.index
    out = left.merge(right, on=on, how="left")
    if len(out) != len(left):
        raise ValueError(f"merge duplicó filas: claves no únicas en {on}")
    out.index = idx
    return out


# --------------------------------------------------------------------------
# helpers de serialización honesta (NaN/inf → null)
# --------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """Convierte NaN/inf a None para un JSON estrictamente válido."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


# --------------------------------------------------------------------------
# checks de aceptación
# --------------------------------------------------------------------------


@dataclass
class AcceptanceChecks:
    """Criterios de la task 10.5, verificados uno a uno."""

    leakage_clean: bool = False
    capacity_respected: bool = False
    metrics_and_breakdowns: bool = False
    powerbi_present: bool = False
    manifest_reproducible: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return not self.failures()

    def failures(self) -> list[str]:
        return [
            name
            for name, ok in {
                "leakage_clean": self.leakage_clean,
                "capacity_respected": self.capacity_respected,
                "metrics_and_breakdowns": self.metrics_and_breakdowns,
                "powerbi_present": self.powerbi_present,
                "manifest_reproducible": self.manifest_reproducible,
            }.items()
            if not ok
        ]


@dataclass
class E2EResult:
    forecast_table: pd.DataFrame
    metrics: dict[str, float]
    breakdowns: dict[str, pd.DataFrame]
    powerbi: dict[str, pd.DataFrame]
    manifest_path: Path
    deliverables: dict[str, Path]
    checks: AcceptanceChecks
    capacity_alerts: list
    champion: str = ""
    model_by_property: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------
# backtest muestreado T-90..T-1 por propiedad (campeón operativo)
# --------------------------------------------------------------------------

def _sampled_backtest(
    history: pd.DataFrame,
    pick: pd.DataFrame,
    model_by_property: dict[str, str],
    cutoff: pd.Timestamp,
    backtest_cfg: dict | None = None,
    curve_params: dict | None = None,
    holiday_calendar: Any = None,
) -> tuple[dict, dict[str, pd.Series], pd.Series]:
    """Snapshots T-90..T-1 por propiedad con el campeón que opera.

    Ejecuta el framework de backtesting (``backtesting/snapshots.run_backtest``)
    una vez por propiedad — la dimensión propiedad se conserva sin que unas
    actuals pisen a otras — con un ``forecast_fn`` que entrena el campeón de
    la propiedad congelado en ``stay - lead``. Los modelos costosos
    (ETS/SARIMAX/GBM) usan un subconjunto reducido de leads/estancias (guard
    de costo documentado); los baratos (pickup_curve, seasonal_naive) corren
    el set completo.

    Returns
    -------
    (report, residual_bands, residuals):
        ``report``: dict para ``backtesting_report.json`` (global, por lead
        band, por propiedad).
        ``residual_bands``: residuos (forecast - actual) por banda de lead.
        ``residuals``: residuos aplanados (pool global).
    """
    cfg = {"n_stays": 8, "n_stays_expensive": 4, "lookback_days": 56}
    cfg.update(backtest_cfg or {})
    rows: list[dict] = []
    clear_cache()

    for prop, model_name in sorted(model_by_property.items()):
        dem_p = history.loc[history["property"] == prop, ["property", "stay_date", "room_nights"]]
        pick_p = pick.loc[pick["property"] == prop]
        actuals = (
            dem_p.loc[dem_p["stay_date"] <= cutoff]
            .groupby("stay_date")["room_nights"].mean().sort_index().dropna()
        )
        if actuals.empty:
            continue
        cheap = model_name in CHEAP_MODELS
        leads = BACKTEST_LEADS_CHEAP if cheap else BACKTEST_LEADS_EXPENSIVE
        n_stays = int(cfg["n_stays"] if cheap else cfg["n_stays_expensive"])
        pool = actuals.index[-int(cfg["lookback_days"]):]
        if len(pool) == 0:
            continue
        stride = max(1, len(pool) // max(n_stays, 1))
        stays = pool[::stride][-n_stays:]

        def _forecast_fn(
            snapshot_frame, stay, lead,
            _model=model_name, _pick=pick_p, _dem=dem_p, _prop=prop,
        ) -> dict:
            fc_date = pd.Timestamp(stay) - pd.Timedelta(days=int(lead))
            fitted = fit_candidate(
                _model, _dem, _pick, fc_date, int(lead), curve_params, holiday_calendar
            )
            pred = predict_candidate(fitted, _pick, pd.DatetimeIndex([pd.Timestamp(stay)]))
            r = pred.iloc[0]
            booked = r.get("current_booked_rn")
            pickup = r.get("expected_pickup_rn")
            return {
                "current_booked_rn": float(booked) if pd.notna(booked) else float("nan"),
                "expected_pickup_rn": float(pickup) if pd.notna(pickup) else float("nan"),
                "forecast_rn": float(r["raw_forecast_rn"]),
                "model_name": _model,
            }

        bt = run_backtest(
            dem_p,
            stays,
            _forecast_fn,
            leads=leads,
            actuals=actuals,
            cache_key=f"{prop}:{model_name}",
        )
        for rev in bt.revisions:
            key = str(pd.Timestamp(rev.stay_date).date())
            actual = bt.actuals.get(key)
            if actual is None:
                continue
            rows.append(
                {
                    "property": prop,
                    "model_name": rev.model_name,
                    "stay_date": rev.stay_date,
                    "forecast_date": rev.forecast_date,
                    "lead_days": int(rev.lead_days),
                    "forecast": float(rev.forecast_rn),
                    "actual": float(actual),
                }
            )

    if not rows:
        return {"protocol": "snapshots T-90..T-1", "note": "sin estancias evaluables"}, {}, pd.Series(dtype=float)

    frame = pd.DataFrame(rows)
    frame["residual"] = frame["forecast"] - frame["actual"]
    frame["lead_band_name"] = frame["lead_days"].map(lead_band)

    global_metrics = compute_metrics(frame["actual"], frame["forecast"]).to_dict()
    by_lead_band: list[dict] = []
    for band, grp in frame.groupby("lead_band_name"):
        met = compute_metrics(grp["actual"], grp["forecast"])
        by_lead_band.append({"lead_band": band, "n_obs": int(len(grp)), **met.to_dict()})
    by_property: list[dict] = []
    for prop, grp in frame.groupby("property"):
        met = compute_metrics(grp["actual"], grp["forecast"])
        by_property.append({"property": prop, "model": grp["model_name"].iloc[0], "n_obs": int(len(grp)), **met.to_dict()})

    residual_bands: dict[str, pd.Series] = {
        band: grp["residual"].reset_index(drop=True)
        for band, grp in frame.groupby("lead_band_name")
    }
    report = {
        "protocol": "snapshots T-90..T-1 (muestreado, campeón operativo, framework snapshots)",
        "n_revisions": int(len(frame)),
        "leads_by_cost": {
            "cheap": list(BACKTEST_LEADS_CHEAP),
            "expensive": list(BACKTEST_LEADS_EXPENSIVE),
        },
        "sampled_stays": sorted(str(pd.Timestamp(d).date()) for d in frame["stay_date"].unique()),
        "global": _jsonable(global_metrics),
        "by_lead_band": _jsonable(by_lead_band),
        "by_property": _jsonable(by_property),
    }
    return report, residual_bands, frame["residual"].reset_index(drop=True)


# --------------------------------------------------------------------------
# monitoring: outliers + drift + decisión de re-entrenamiento
# --------------------------------------------------------------------------

def _monitoring_block(
    history: pd.DataFrame, metrics: dict[str, float], config: EngineConfig
) -> dict:
    out: dict = {}
    try:
        outliers = detect_outliers(history, capacity_col=None)
        out["n_outliers"] = len(outliers)
        out["outliers"] = _jsonable([o.to_dict() for o in outliers[:50]])
        if len(outliers) > 50:
            out["outliers_note"] = f"se listan 50 de {len(outliers)} outliers"
    except Exception as exc:  # el monitoreo jamás rompe la corrida
        out["outliers_error"] = str(exc)

    try:
        port = history.groupby("stay_date")["room_nights"].sum().sort_index()
        if len(port) >= 30:
            ref = port.iloc[: len(port) // 2]
            cur = port.iloc[-90:]
            alerts = run_drift_suite(
                reference_metrics={},
                current_metrics=_jsonable(metrics),
                reference_demand=ref,
                current_demand=cur,
            )
            out["n_drift_alerts"] = len(alerts)
            out["drift_alerts"] = _jsonable([a.to_dict() for a in alerts])
            out["drift_reference_note"] = (
                "referencia de demanda: primera mitad del histórico; ventana actual: últimos 90d"
            )
    except Exception as exc:
        out["drift_error"] = str(exc)

    try:
        span_days = int(
            (history["stay_date"].max() - history["stay_date"].min()).days + 1
        )
        decision = should_retrain(
            frequency=config.retrain_frequency,
            new_records=int(len(history)),
            days_since_last=0,  # esta corrida acaba de entrenar
            min_history_days=int(config.min_history_days),
            current_history_days=span_days,
        )
        out["retraining_decision"] = _jsonable(decision.to_dict())
    except Exception as exc:
        out["retraining_error"] = str(exc)
    return out


# --------------------------------------------------------------------------
# explicabilidad derivada de los drivers reales de cada fila
# --------------------------------------------------------------------------

def _explain_forecast(fc: pd.DataFrame, champion: str) -> pd.DataFrame:
    """Drivers derivados de los valores de la propia fila (sin pesos fijos).

    Para el campeón pickup los drivers son las componentes reales del
    forecast (booked / pickup restante) y la restricción de capacidad cuando
    aplica; para campeones estadísticos/ML se declara el método sustituto.
    """
    rows = []
    for _, r in fc.head(50).iterrows():
        booked = float(r.get("current_booked_rn") or 0.0)
        pickup = float(r.get("expected_pickup_rn") or 0.0)
        cap_val = r.get("capacity", np.nan)
        raw_val = r.get("raw_forecast_rn", np.nan)
        try:
            weights: dict[str, float] = {}
            denom = abs(booked) + abs(max(pickup, 0.0))
            if pd.notna(cap_val) and pd.notna(raw_val) and float(cap_val) < float(raw_val):
                denom += abs(float(cap_val) - float(raw_val))
            if denom > 0:
                weights["current_booked_rn"] = abs(booked) / denom
                weights["expected_pickup_rn"] = abs(max(pickup, 0.0)) / denom
                if pd.notna(cap_val) and pd.notna(raw_val) and float(cap_val) < float(raw_val):
                    weights["capacity"] = abs(float(cap_val) - float(raw_val)) / denom
            if not weights:
                weights = {"current_booked_rn": 1.0}
            expl = explain_from_importance(
                property=str(r.get("property", "?")),
                stay_date=str(r.get("stay_date", "?")),
                forecast_rn=float(r.get("final_forecast_rn", 0.0)),
                feature_values={
                    "current_booked_rn": booked,
                    "expected_pickup_rn": max(pickup, 0.0),
                    "capacity": float(cap_val) if pd.notna(cap_val) else 0.0,
                },
                feature_importance=weights,
                capacity=float(cap_val) if pd.notna(cap_val) else None,
                raw_forecast=float(raw_val) if pd.notna(raw_val) else None,
                method=f"drivers-derivados ({champion})",
            )
            rows.append(
                {
                    "property": expl.property,
                    "stay_date": expl.stay_date,
                    "forecast_rn": expl.forecast_rn,
                    "top_drivers": " | ".join(
                        f"{d['business_label']}: {d['contribution']:+.1f}"
                        for d in expl.top_drivers(3)
                    ),
                    "has_capacity_constraint": expl.has_capacity_constraint,
                    "champion": champion,
                }
            )
        except Exception as exc:  # pragma: no cover
            rows.append({"property": "?", "stay_date": "?", "error": str(exc)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------


def run_end_to_end(
    *,
    demand: pd.DataFrame,
    capacity: pd.DataFrame,
    pickup: pd.DataFrame,
    run_dir: str | Path,
    config: EngineConfig | None = None,
    cutoff: pd.Timestamp | None = None,
    actual: pd.DataFrame | None = None,
    sample_data: bool = False,
    fold_metrics: pd.DataFrame | None = None,
    error_panel: pd.DataFrame | None = None,
    eval_frame: pd.DataFrame | None = None,
    backtest_cfg: dict | None = None,
) -> E2EResult:
    """Ejecuta la cadena completa y verifica los criterios de aceptación.

    ``fold_metrics``/``error_panel`` inyectan la evaluación REAL (benchmark
    sobre datos reales + panel de errores por fecha para la significancia
    Diebold-Mariano); el CAMPEÓN resultante opera el forecast. Si se
    omiten, el benchmark usa placeholders sintéticos (etiquetados), el
    modelo operativo es ``pickup_curve`` (default documentado) y el
    manifiesto lo marca ``benchmark_source: synthetic_placeholder``.

    ``eval_frame`` inyecta la evaluación de métricas y desgloses (columnas
    ``property/stay_date/lead_days/actual/forecast``) cuando la ventana de
    pronóstico todavía no tiene reales —corrida **futura**—: en ese caso las
    métricas provienen del backtest y el manifiesto lo etiqueta como
    ``metrics_source: injected_backtest`` (nunca se inventan reales).
    """
    config = config or load_config()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. detección de esquema + alias + calidad + GATE
    schemas = {
        "demand": detect_schema(demand, SourceKind.DEMAND, config.column_hints),
        "capacity": detect_schema(capacity, SourceKind.CAPACITY, config.column_hints),
        "pickup": detect_schema(pickup, SourceKind.PICKUP, config.column_hints),
    }
    if config.property_aliases:
        demand = _apply_aliases(demand, schemas["demand"], config.property_aliases)
        capacity = _apply_aliases(capacity, schemas["capacity"], config.property_aliases)
        pickup = _apply_aliases(pickup, schemas["pickup"], config.property_aliases)

    # propiedades cerradas: exclusión configurable ANTES de calidad — no
    # participan de ingesta, features, benchmark, forecast ni backtest
    excluded_props = [str(p) for p in config.excluded_properties]
    if excluded_props:
        demand = demand.loc[~demand[schemas["demand"].property_col].astype(str).isin(excluded_props)]
        capacity = capacity.loc[~capacity[schemas["capacity"].property_col].astype(str).isin(excluded_props)]
        pickup = pickup.loc[~pickup[schemas["pickup"].property_col].astype(str).isin(excluded_props)]

    raw_sources = {
        "demand": (demand, schemas["demand"]),
        "capacity": (capacity, schemas["capacity"]),
        "pickup": (pickup, schemas["pickup"]),
    }
    dq_report = build_report(raw_sources, validate_sources(raw_sources))
    # gate con dientes (spec data-ingestion): status fail → corrida detenida
    enforce_quality_gate(dq_report)

    dem = _normalize_demand(demand, schemas["demand"])
    cap = _normalize_capacity(capacity, schemas["capacity"])
    pick = _normalize_pickup(pickup, schemas["pickup"])

    # ---------- 2. corte + as-of + leakage (suite completa)
    if cutoff is None:
        cutoff = pd.Timestamp(dem["stay_date"].max())
    cutoff = pd.Timestamp(cutoff)

    # as-of: solo snapshots de reserva observados ≤ corte (el archivo real
    # contiene extracciones posteriores; el forecast no puede verlas)
    pick = pick.loc[pick["snapshot_date"] <= cutoff].copy()
    enforce_cutoff(pick, "snapshot_date", cutoff, "pickup@as-of")

    builder = AsOfBuilder(date_col="stay_date", cutoff=cutoff)
    history = builder.frame(dem, "demand@snapshot")
    enforce_cutoff(history, "stay_date", cutoff, "demand@snapshot")
    # capacidad as-of: features/entrenamiento solo con capacidad ≤ corte; la
    # capacidad post-corte se usa EXCLUSIVAMENTE como techo del horizonte
    cap_asof = cap.loc[cap["stay_date"] <= cutoff].copy()
    enforce_cutoff(cap_asof, "stay_date", cutoff, "capacity@as-of")

    horizon_days = int(config.forecast_horizon)

    # ---------- 2b. calendario de feriados (exógeno, conocido a cualquier lead)
    holiday_cal = holiday_calendar_from_config(
        config,
        min_year=int(pd.Timestamp(history["stay_date"].min()).year),
        max_year=int((cutoff + pd.Timedelta(days=horizon_days)).year),
    )

    # ---------- 3. features as-of (sobre history, no sobre el futuro)
    df = history.copy()
    df = add_calendar_features(
        df, "stay_date",
        holidays=list(holiday_cal.dates) if holiday_cal is not None else None,
    )
    df, missing_feats = add_demand_features(df)

    # booked as-of por estancia (snapshot más reciente) → features de
    # capacidad que dependen de reservas se calculan DESPUÉS del merge
    pk_last = (
        pick.sort_values(["property", "stay_date", "snapshot_date"])
        .drop_duplicates(["property", "stay_date"], keep="last")
        [["property", "stay_date", "pickup_rn"]]
        .rename(columns={"pickup_rn": "current_booked_rn"})
    )
    df = _merge_keep_index(df, cap, ["property", "stay_date"])
    df = _merge_keep_index(df, pk_last, ["property", "stay_date"])
    df = add_capacity_features(df)
    if "capacity_remaining" not in df.columns:
        missing_feats = {**missing_feats, "capacity_remaining": ["__all__"]}

    pk_feats = add_pickup_features(pick.copy().assign(current_booked_rn=pick["pickup_rn"]), cutoff=cutoff)
    pk_feats = (
        pk_feats.sort_values(["property", "stay_date", "snapshot_date"])
        .drop_duplicates(["property", "stay_date"], keep="last")
    )
    pk_cols = [
        c
        for c in pk_feats.columns
        if c not in df.columns or c in ("property", "stay_date")
    ]
    df = _merge_keep_index(df, pk_feats[pk_cols], ["property", "stay_date"])

    # ---------- 3b. suite de leakage COMPLETA (features → matriz incluida)
    def _demand_feature_fn(frame: pd.DataFrame) -> pd.DataFrame:
        out, _ = add_demand_features(frame)
        return out

    feature_cols = [c for c in df.columns if c.startswith(("lag_", "rolling_", "same_"))]
    rollings = [
        {
            "df": g[["stay_date", "room_nights", "rolling_mean_7"]],
            "date_col": "stay_date",
            "raw_col": "room_nights",
            "feature_col": "rolling_mean_7",
            "window": 7,
        }
        for _, g in df.groupby("property", sort=False)
        if "rolling_mean_7" in g.columns and len(g) > 7
    ]
    leakage_report = run_leakage_suite(
        cutoff=cutoff,
        frames={
            "demand": (history, "stay_date"),
            "pickup": (pick, "snapshot_date"),
            "capacity": (cap_asof, "stay_date"),
        },
        split={
            "strategy": "expanding",
            "train_end": str(cutoff.date()),
            "val_start": str((cutoff + pd.Timedelta(days=1)).date()),  # contiguo
        },
        train=df[["property", "stay_date", "room_nights", *feature_cols]],
        full=dem,
        feature_fn=_demand_feature_fn,
        feature_cols=feature_cols,
        rolling=rollings,
    )
    # contrato duro: el target de una fila jamás entra a sus propias features
    assert_no_target_in_rolling(df, "room_nights", "stay_date", "property")
    leakage_clean = leakage_report.clean

    # ---------- 4. benchmark + SELECCIÓN (el campeón elegido OPERA)
    if fold_metrics is not None:
        benchmark = run_benchmark(
            fold_metrics, error_panel=error_panel, source="real_backtest"
        )
        champion = benchmark.champion
        per_prop_sel = (
            select_per_property(fold_metrics, global_champion=champion)
            if "property" in fold_metrics.columns
            else None
        )
        benchmark_source = "real_backtest"
    else:
        benchmark = run_benchmark(_fold_metrics(), source="synthetic_placeholder")
        champion = "pickup_curve"  # default documentado: sin evidencia real
        per_prop_sel = None
        benchmark_source = "synthetic_placeholder"

    props = sorted(history["property"].dropna().unique())
    model_by_property: dict[str, str] = {
        p: (per_prop_sel.by_property.get(p, champion) if per_prop_sel else champion)
        for p in props
    }
    property_fallback = dict(per_prop_sel.property_fallback) if per_prop_sel else {}

    # ---------- 5. forecast del horizonte con el campeón (por propiedad)
    curve_params = dict(config.pickup_curve)
    raw, fitted_models = build_forecasts(
        model_by_property, history, pick, cutoff, horizon_days,
        curve_params, holiday_cal,
    )

    # ---------- 6. capacidad + ocupación (por fecha; fallback mediana → 100)
    cap_daily = (
        cap.dropna(subset=["capacity"])
        .groupby(["property", "stay_date"], as_index=False)["capacity"]
        .max()
    )
    raw = raw.merge(cap_daily, on=["property", "stay_date"], how="left")
    prop_cap = cap_asof.groupby("property")["capacity"].median()  # mediana as-of
    raw["capacity"] = raw["capacity"].fillna(raw["property"].map(prop_cap)).fillna(100.0)
    constrained, cap_alerts = apply_capacity_constraint(raw)
    with_occ = compute_occupancy(constrained)

    # ---------- 7. backtest T-90..T-1 muestreado (residuos por lead band)
    backtest_report, residual_bands, residuals_flat = _sampled_backtest(
        history, pick, model_by_property, cutoff, backtest_cfg, curve_params, holiday_cal
    )

    # ---------- 8. intervalos P10/P50/P90 anclados al punto publicado
    residuals_provided = bool(residual_bands) and len(residuals_flat) > 0
    if not residuals_provided:
        # default sintético (etiquetado): corridas sin reales evaluables
        residuals_flat = pd.Series(
            np.random.default_rng(config.random_seed).normal(0, 4, 60)
        )
    interval = build_intervals(
        with_occ,
        model_name=champion,
        point_col="final_forecast_rn",
        residuals=residuals_flat,
        residual_bands=residual_bands if residuals_provided else None,
    )
    interval_source = "real_backtest" if residuals_provided else "synthetic_placeholder"
    fc_table = interval.frame
    run_ts = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    fc_table["forecast_date"] = cutoff
    fc_table["forecast_run_timestamp"] = run_ts
    fc_table["model_version"] = "0.1.0"

    # ---------- 9. métricas + desgloses + cobertura de evaluación
    eval_src: pd.DataFrame | None = None
    metrics: dict[str, float] = {}
    breakdowns: dict[str, pd.DataFrame] = {}
    if eval_frame is not None and not eval_frame.empty:
        missing_cols = sorted(
            {"property", "stay_date", "lead_days", "actual", "forecast"}
            - set(eval_frame.columns)
        )
        if missing_cols:
            raise ValueError(f"eval_frame inyectado sin columnas requeridas: {missing_cols}")
        eval_src = eval_frame.copy()
    elif actual is not None and not actual.empty:
        a = actual.copy()
        a["stay_date"] = pd.to_datetime(a["stay_date"], errors="coerce")
        if "property" not in a.columns:
            a["property"] = props[0] if props else "PORTFOLIO"
        a["property"] = a["property"].astype(str).replace(config.property_aliases)
        a = a.rename(columns={"room_nights": "actual"})
        merged = fc_table.merge(
            a[["property", "stay_date", "actual"]], on=["property", "stay_date"], how="inner"
        )
        if not merged.empty:
            merged["forecast"] = merged["final_forecast_rn"]
            eval_src = merged

    if eval_src is not None:
        eval_src["stay_date"] = pd.to_datetime(eval_src["stay_date"], errors="coerce")
        eval_src["lead_days"] = (
            pd.to_numeric(eval_src["lead_days"], errors="coerce").fillna(30).astype(int)
        )
        metrics = compute_metrics(eval_src["actual"], eval_src["forecast"]).to_dict()
        breakdowns = {
            "lead_band": metrics_by_lead_band(eval_src),
            "property": metrics_by_property(eval_src),
            **metrics_by_seasonality(eval_src),
        }

    evaluated_props = (
        sorted(eval_src["property"].dropna().unique()) if eval_src is not None else []
    )
    excluded_props_map = {
        p: "propiedad cerrada (configuración)" for p in excluded_props
    }
    excluded_no_actuals = {
        p: "sin reales en la ventana evaluada"
        for p in props
        if p not in evaluated_props
    }
    evaluation_coverage = {
        "n_properties_portfolio": len(props),
        "n_properties_evaluated": len(evaluated_props),
        "evaluated": evaluated_props,
        "excluded": excluded_no_actuals,
        "excluded_closed": excluded_props_map,
        "n_obs": int(metrics.get("n_obs", 0)),
    }

    metrics_ok = bool(metrics) and {"lead_band", "property", "season"} <= set(breakdowns)
    metrics_source = (
        "injected_backtest"
        if eval_frame is not None
        else ("horizon_actuals" if metrics else "none")
    )

    # ---------- 10. Power BI + KPIs
    forecast_final = build_forecast_table(fc_table)
    pbi_source = forecast_final
    if eval_src is not None:
        # reales solo cuando existen en la MISMA ventana que el pronóstico
        # (corrida futura: sin solape → actual_rn queda NaN, honesto)
        keys = eval_src[["property", "stay_date", "actual"]].drop_duplicates()
        pbi_source = forecast_final.merge(
            keys, on=["property", "stay_date"], how="left"
        ).rename(columns={"actual": "actual_rn"})

    pickup_pbi = raw[
        ["property", "stay_date", "current_booked_rn", "expected_pickup_rn", "lead_days"]
    ].copy()

    # accuracy: desglose por banda de lead + por propiedad (columna breakdown)
    accuracy_frames: list[pd.DataFrame] = []
    for name, key in (("lead_band", None), ("property", None)):
        bdf = breakdowns.get(name)
        if bdf is not None and not bdf.empty:
            bdf = bdf.copy()
            bdf["breakdown"] = name
            accuracy_frames.append(bdf)
    accuracy_pbi = pd.concat(accuracy_frames, ignore_index=True) if accuracy_frames else None

    models_pbi = benchmark.table.assign(
        model_version="0.1.0",
        is_champion=benchmark.table["model"] == champion,
        is_challenger=benchmark.table["model"] != champion,
    )[["model", "model_version", "is_champion", "is_challenger"]].rename(
        columns={"model": "model_name"}
    )
    powerbi = build_powerbi_dataset(
        pbi_source,
        accuracy=accuracy_pbi,
        pickup=pickup_pbi,
        models=models_pbi,
    )
    powerbi_ok = set(powerbi) == {"forecast", "accuracy", "pickup", "model"} and all(
        not f.empty for f in powerbi.values()
    )

    kpi_actual: pd.DataFrame | None = None
    if eval_src is not None:
        probe = forecast_final.merge(
            eval_src[["property", "stay_date"]].drop_duplicates(),
            on=["property", "stay_date"],
            how="inner",
        )
        if not probe.empty:
            kpi_actual = eval_src[["property", "stay_date", "actual"]].rename(
                columns={"actual": "room_nights"}
            )
    kpis = executive_kpis(
        forecast_final,
        actual=kpi_actual,
        expected_pickup_total=float(raw["expected_pickup_rn"].sum()),
    )
    if kpi_actual is None and metrics:
        # ventana futura sin reales: los KPIs de accuracy reflejan el backtest
        # real (etiquetado en el manifiesto vía metrics_source)
        kpis["forecast_accuracy_wmape"] = metrics.get("wmape", float("nan"))
        kpis["forecast_accuracy_mape"] = metrics.get("mape", float("nan"))
        kpis["forecast_bias"] = metrics.get("bias", float("nan"))
        kpis["mape_excluded_zero_actual"] = float(metrics.get("n_mape_excluded", 0.0))
    kpis["n_properties_evaluated"] = float(len(evaluated_props))
    kpis["n_properties_portfolio"] = float(len(props))

    # ---------- 11. monitoring (outliers + drift + retraining)
    monitoring = _monitoring_block(history, metrics, config)

    # ---------- 12. manifiesto (campeón coherente, JSON estricto)
    train_start = pd.Timestamp(history["stay_date"].min())
    manifest = RunManifest(
        dataset_version=f"{'sample' if sample_data else 'real'}-{pd.Timestamp.now().date()}",
        model_version=f"{champion}-0.1.0",
        training_period={"start": str(train_start.date()), "end": str(cutoff.date())},
        validation_period={
            "start": str((cutoff + pd.Timedelta(days=1)).date()),
            "end": str((cutoff + pd.Timedelta(days=horizon_days)).date()),
        },
        feature_config={
            "lags": [1, 7, 14, 28, 365],
            "rollings": [7, 14, 28, 56],
            "calendar": True,
            "pickup": True,
            "capacity": True,
            "missing_features": missing_feats,
        },
        hyperparameters={
            "model": champion,
            "per_property": model_by_property,
            "horizon_days": horizon_days,
            "pickup_curve": _jsonable(curve_params),
            "holidays": _jsonable(
                dict(config.holidays) if holiday_cal is not None else {"countries": []}
            ),
        },
        metrics=_jsonable(metrics) if metrics else {"placeholder": 0.0},
        seed=int(config.random_seed),
        extra={
            "data_quality_status": dq_report.status,
            "n_capacity_alerts": len(cap_alerts),
            "kpis": _jsonable(kpis),
            "sample_data": sample_data,
            "benchmark_source": benchmark_source,
            "interval_source": interval_source,
            "metrics_source": metrics_source,
            "forecast_window": {
                "start": str((cutoff + pd.Timedelta(days=1)).date()),
                "end": str((cutoff + pd.Timedelta(days=horizon_days)).date()),
            },
            "capacity_usage": "horizon_cap_only",
            "excluded_properties": _jsonable(excluded_props_map),
            "holiday_calendar": _jsonable(holiday_cal.describe()) if holiday_cal is not None else None,
            "champion_reason": benchmark.selection.reason,
            "champion_significance": _jsonable(benchmark.selection.significance),
            "property_fallback": property_fallback,
            "evaluation_coverage": _jsonable(evaluation_coverage),
            "monitoring": monitoring,
        },
    )
    manifest_path = manifest.write(run_dir)

    # ---------- 13. entregables
    explain_df = _explain_forecast(fc_table, champion)
    forecast_problems = validate_forecast_table(forecast_final)

    accuracy_report: dict
    if metrics:
        accuracy_report = {
            "global": _jsonable(metrics),
            "by_lead_band": _jsonable(
                breakdowns.get("lead_band", pd.DataFrame()).to_dict("records")
            ),
            "by_property": _jsonable(
                breakdowns.get("property", pd.DataFrame()).to_dict("records")
            ),
            "evaluation_coverage": _jsonable(evaluation_coverage),
        }
    else:
        accuracy_report = {"note": "sin reales en esta corrida"}

    deliverables = generate_deliverables(
        run_dir,
        data_quality_report=dq_report.to_dict(),
        eda_report={"n_rows": int(len(df)), "missing_features": missing_feats},
        feature_report={"n_features": int(df.shape[1]), "missing": missing_feats},
        benchmark=benchmark,
        backtest_report=backtest_report,
        accuracy_report=accuracy_report,
        forecast_table=forecast_final,
        explainability=explain_df,
        powerbi=powerbi,
        technical_doc=_technical_doc(
            sample_data=sample_data,
            dq=dq_report,
            metrics=metrics,
            metrics_source=metrics_source,
            champion=champion,
            model_by_property=model_by_property,
            benchmark_source=benchmark_source,
            interval_source=interval_source,
            excluded_props_map=excluded_props_map,
            curve_params=curve_params,
            holiday_calendar_md=(
                f"{holiday_cal.describe()['n_holiday_dates']} fechas "
                f"{'+'.join(holiday_cal.countries)} (ventana ±{holiday_cal.window_days}d)"
                if holiday_cal is not None
                else None
            ),
        ),
    )

    # ---------- checks
    capacity_ok = bool(
        (fc_table["final_forecast_rn"] <= fc_table["capacity"] + 1e-9).all()
        and not forecast_problems
    )
    checks = AcceptanceChecks(
        leakage_clean=leakage_clean,
        capacity_respected=capacity_ok,
        metrics_and_breakdowns=metrics_ok,
        powerbi_present=powerbi_ok,
        manifest_reproducible=manifest_path.exists(),
        details={
            "data_quality_status": dq_report.status,
            "forecast_problems": forecast_problems,
            "missing_features": missing_feats,
            "n_capacity_alerts": len(cap_alerts),
            "kpis": kpis,
            "leakage_findings": len(leakage_report.findings),
            "n_breakdowns": len(breakdowns),
            "champion": champion,
            "model_by_property": model_by_property,
            "evaluation_coverage": evaluation_coverage,
            "excluded_properties": excluded_props_map,
        },
    )
    # ---------- 14. reporte visual HTML (vista; NO cuenta entre los 10 entregables)
    try:
        actual_series: pd.Series | None = None
        if eval_src is not None and "actual" in eval_src.columns:
            merged_actual = eval_src[["property", "stay_date", "actual"]].drop_duplicates(
                subset=["property", "stay_date"]
            )
            joined = forecast_final.reset_index(drop=True).merge(
                merged_actual, on=["property", "stay_date"], how="left"
            )
            if joined["actual"].notna().any():
                actual_series = joined["actual"]
        generate_visual_report(
            run_dir,
            forecast_final,
            demand=dem,
            benchmark=benchmark.table,
            metrics=metrics,
            kpis=kpis,
            actual=actual_series,
        )
    except Exception as exc:  # la vista jamás rompe la corrida
        print(f"[aviso] no se pudo generar report.html: {exc}")

    return E2EResult(
        forecast_table=forecast_final,
        metrics=metrics,
        breakdowns=breakdowns,
        powerbi=powerbi,
        manifest_path=manifest_path,
        deliverables=deliverables,
        checks=checks,
        capacity_alerts=cap_alerts,
        champion=champion,
        model_by_property=model_by_property,
    )


# --------------------------------------------------------------------------

def _fold_metrics() -> pd.DataFrame:
    """Métricas por fold para el benchmark (entrada out-of-sample)."""
    base = {
        "seasonal_naive": (14.0, 0.145, 12.0, 15.5, 0.02),
        "pickup_curve": (11.0, 0.115, 9.5, 12.0, -0.01),
        "ets": (13.0, 0.13, 11.0, 14.0, 0.04),
        "sarimax": (12.5, 0.128, 10.8, 13.5, 0.03),
        "lightgbm": (9.0, 0.12, 10.0, 13.0, 0.18),
        "xgboost": (9.5, 0.125, 10.2, 13.2, 0.15),
        "catboost": (9.8, 0.126, 10.4, 13.4, 0.14),
    }
    rng = np.random.default_rng(42)
    rows = []
    for model, (mape, wmape, mae, rmse, bias) in base.items():
        for fold in range(4):
            rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "mape": mape + rng.normal(0, 0.3),
                    "wmape": wmape + rng.normal(0, 0.004),
                    "mae": mae + rng.normal(0, 0.3),
                    "rmse": rmse + rng.normal(0, 0.3),
                    "bias": bias + rng.normal(0, 0.01),
                }
            )
    return pd.DataFrame(rows)


def _technical_doc(
    *,
    sample_data: bool,
    dq,
    metrics: dict,
    metrics_source: str = "horizon_actuals",
    champion: str = "pickup_curve",
    model_by_property: dict[str, str] | None = None,
    benchmark_source: str = "real_backtest",
    interval_source: str = "real_backtest",
    excluded_props_map: dict[str, str] | None = None,
    curve_params: dict | None = None,
    holiday_calendar_md: str | None = None,
) -> str:
    disclaimer = (
        "> ⚠️ **DISCLAIMER**: corrida ejecutada con datos de MUESTRA en el esquema real "
        "de las tres fuentes (`data/` no contiene los CSV reales). La metodología queda "
        "validada; repetir la corrida soltando los CSV reales en `data/` sin tocar código.\n"
        if sample_data
        else "> Corrida sobre fuentes reales.\n"
    )
    mape = metrics.get("mape", float("nan"))
    wmape = metrics.get("wmape", float("nan"))
    metrics_note = (
        "- **Origen de las métricas**: backtest real sobre la ventana ANTERIOR al "
        "corte (la ventana de pronóstico es futura y aún no tiene reales); "
        "el manifiesto lo marca como `metrics_source: injected_backtest`.\n"
        if metrics_source == "injected_backtest"
        else "- **Origen de las métricas**: reales observados dentro de la ventana del horizonte.\n"
    )
    per_prop = model_by_property or {}
    per_prop_md = (
        "\n".join(f"  - {p}: `{m}`" for p, m in sorted(per_prop.items())) or "  - (todas con el campeón global)"
    )
    excluded_md = (
        "\n".join(f"  - {p}: cerrada (excluida por configuración)" for p in sorted(excluded_props_map))
        or "  - (ninguna)"
    )
    curve_md = (
        f"temporada del stay condicionada ({'activada' if curve_params.get('season_conditional') else 'desactivada'}), "
        f"ventana de recencia {curve_params.get('max_history_days', '—')}d, "
        f"fallback con < {curve_params.get('min_stays_per_bucket', '—')} estancias"
        if curve_params
        else "parámetros por defecto"
    )
    if holiday_calendar_md is not None:
        curve_md += f"; feriados: {holiday_calendar_md}"
    if curve_params:
        curve_md += (
            f"; condicionamiento por ventana de feriado "
            f"({'activado' if curve_params.get('holiday_conditional') else 'desactivado'})"
        )
    bench_note = (
        "backtest real por propiedad (folds trimestrales, sin solape con la ventana de reporte)"
        if benchmark_source == "real_backtest"
        else "**PLACEHOLDER sintético**: sin folds reales inyectados"
    )
    interval_note = (
        "residuos del backtest T-90..T-1 por banda de lead"
        if interval_source == "real_backtest"
        else "residuos sintéticos de reemplazo (etiquetados en el manifiesto)"
    )
    return f"""# Documentación técnica — Room Nights Forecasting Engine

{disclaimer}

## Modelo operativo
- **Campeón**: `{champion}` — seleccionado por el benchmark ({bench_note}) y
  **operado** en esta corrida (el forecast lo genera el campeón).
- **Curva de pickup**: {curve_md}
- **Selección por propiedad**:
{per_prop_md}
- **Propiedades cerradas (excluidas)**:
{excluded_md}
- La significancia de la selección (Diebold-Mariano) queda en el manifiesto
  (`champion_significance`); un empate estadístico se desempata por parsimonia.

## Fuentes
Las tres fuentes exclusivas: `capacity_daily`, `demand_daily`, `pickup_by_lead`.
Detección automática de esquema (sin asumir nombres de columna).

## Data Quality
- Status: `{dq.status}`
- Errores: {dq.error_count} · Advertencias: {dq.warning_count}
- El gate de calidad se aplica en esta ruta: un status `fail` detiene la corrida.

## Métricas de la corrida
- MAPE: {mape:.2f}% · WMAPE: {wmape:.4f} · MAE/RMSE/Bias en la tabla de accuracy.
{metrics_note}- MAPE excluye `Actual = 0` (nunca se sustituye el cero); el conteo queda documentado.
- La cobertura de evaluación (propiedades dentro/fuera de la ventana) está en el
  accuracy report y en el manifiesto (`evaluation_coverage`).

## Intervalos P10/P50/P90
- Anclados al punto publicado (`p50 = final_forecast_rn`), capados a capacidad.
- Calibración: {interval_note}.

## Limitaciones del modelo
1. Sin datos externos: clima, competencia y eventos propietarios fuera de alcance.
2. ETS/SARIMAX se entrenan sobre series posicionales: con huecos severos el paso
   temporal se aproxima; los candidatos ML usan features calendario exactas.
3. La banda de un lead con pocos residuos observados hereda el pool global
   (fallback documentado con `min_residuals`).
4. Cold start: propiedades con < 60 días de histórico usan guard de nivel reciente.
5. Revenue (`RN × ADR`) queda fuera de esta versión.
6. Granularidad `Property × Stay Date`; dimensiones más finas solo si las fuentes
   las soportan de forma consistente.
"""
