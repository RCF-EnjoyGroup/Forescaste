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

from src.data.dates import parse_dates_flexible

from src.backtesting.snapshots import clear_cache, run_backtest
from src.config import EngineConfig, load_config
from src.data.schema import SourceKind, SourceSchema, detect_schema
from src.evaluation.breakdowns import (
    interval_coverage,
    lead_band,
    metrics_by_lead_band,
    metrics_by_property,
    metrics_by_property_weekly,
    metrics_by_seasonality,
)
from src.evaluation.metrics import compute_metrics
from src.evaluation.selection import select_per_property
from src.explainability.explainer import explain_from_importance
from src.features.asof import AsOfBuilder, enforce_cutoff
from src.features.calendar import add_calendar_features
from src.features.capacity import add_capacity_features
from src.features.demand import add_demand_features
from src.features.events import load_events
from src.features.holidays import (
    HolidayCalendar,
    calendar_with_events,
    holiday_calendar_from_config,
)
from src.features.pickup import add_pickup_features
from src.forecasting.constraints import (
    BookedFloorAlert,
    apply_capacity_constraint,
    build_intervals,
    compute_occupancy,
)
from src.forecasting.core import (  # re-exportados para tests/CLI
    BACKTEST_LEADS_CHEAP,
    BACKTEST_LEADS_EXPENSIVE,
    CHEAP_MODELS,
    FittedCandidate,
    ML_FACTORIES,
    _apply_aliases,
    _booked_requests,
    _normalize_capacity,
    _normalize_demand,
    _normalize_pickup,
    _remaining_pickup_curves,
    apply_bias_factors,
    apply_level_factors,
    build_forecasts,
    fit_candidate,
    lead_bias_factors,
    level_correction_factors,
    operative_residual_bands,
    predict_candidate,
)
from src.monitoring.accuracy_tracking import (
    build_track_record,
    forecast_revision,
    load_published_forecast,
    score_vintage,
    select_gradable_publication,
    vintage_detail,
)
from src.monitoring.drift import (
    DriftAlert,
    demand_drift_alerts,
    detect_outliers,
    run_drift_suite,
    should_retrain,
)
from src.monitoring.run_manifest import RunManifest
from src.monitoring.run_registry import (
    RunRecord,
    append_record,
    latest_record,
    load_registry,
)
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
        return {
            (str(k) if isinstance(k, tuple) else k): _jsonable(v)
            for k, v in obj.items()
        }
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

    # convergencia por propiedad: cómo decae el error conforme se acerca la
    # estancia (una observación por propiedad × banda de lead) — base de la
    # gráfica de revisiones del report.html
    by_property_lead: list[dict] = []
    for (prop, band), grp in frame.groupby(["property", "lead_band_name"]):
        met = compute_metrics(grp["actual"], grp["forecast"])
        by_property_lead.append(
            {"property": prop, "lead_band": band, "n_obs": int(len(grp)), **met.to_dict()}
        )

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
        "by_property_lead_band": _jsonable(by_property_lead),
    }
    return report, residual_bands, frame["residual"].reset_index(drop=True)


# --------------------------------------------------------------------------
# monitoring: outliers + drift + decisión de re-entrenamiento
# --------------------------------------------------------------------------

def _monitoring_block(
    history: pd.DataFrame,
    metrics: dict[str, float],
    config: EngineConfig,
    *,
    cutoff: pd.Timestamp,
    metrics_source: str = "",
    previous: dict[str, Any] | None = None,
    coverage: dict[str, Any] | None = None,
) -> dict:
    """Monitoreo con estado real del registro de corridas.

    - Outliers con baseline LOCAL rodante (la media/σ de todo el histórico
      flaggeaba el propio arranque de las propiedades).
    - Drift de demanda POR PROPIEDAD contra la misma ventana YoY (PSI) —
      la referencia "primera mitad del histórico" disparaba SIEMPRE.
    - Drift de métricas (10.1) contra la corrida anterior registrada.
    - Re-entrenamiento (10.4) con ``days_since_last``/``new_records``
      reales del registro (antes: hardcodeado 0 y todo el histórico).
    - Desviación de cobertura de banda por propiedad (P10–P90 nominal 80%).
    """
    out: dict = {}
    mon_cfg = dict(config.monitoring or {})

    try:
        outliers = detect_outliers(history, capacity_col=None)
        out["n_outliers"] = len(outliers)
        out["outliers"] = _jsonable([o.to_dict() for o in outliers[:50]])
        if len(outliers) > 50:
            out["outliers_note"] = f"se listan 50 de {len(outliers)} outliers"
    except Exception as exc:  # el monitoreo jamás rompe la corrida
        out["outliers_error"] = str(exc)

    alerts: list[DriftAlert] = []
    try:
        demand_alerts, skipped = demand_drift_alerts(
            history,
            cutoff=cutoff,
            current_window_days=int(mon_cfg.get("drift_current_window_days", 90)),
            psi_warning=float(mon_cfg.get("psi_warning", 0.30)),
            psi_alert=float(mon_cfg.get("psi_alert", 0.50)),
        )
        alerts.extend(demand_alerts)
        out["drift_skipped_properties"] = _jsonable(skipped)
        out["drift_reference_note"] = (
            "referencia de drift: misma ventana calendario un año atrás "
            "(YoY), por propiedad (PSI) — la primera mitad del histórico "
            "incluía la era de 1-2 hoteles y disparaba en cada corrida"
        )
    except Exception as exc:  # pragma: no cover - guard defensivo
        out["drift_error"] = str(exc)

    # 10.1 con estado: comparación REAL contra la corrida anterior registrada
    # (solo con fuente de métricas comparable: verificación vs backtest
    # inyectado miden ventanas distintas — manzanas vs naranjas)
    previous = previous or {}
    prev_metrics = previous.get("metrics") or {}
    try:
        if prev_metrics and str(previous.get("metrics_source", "")) == metrics_source:
            alerts.extend(
                run_drift_suite(
                    reference_metrics=prev_metrics,
                    current_metrics=_jsonable(metrics),
                )
            )
            out["metric_drift_reference"] = {
                "run_id": previous.get("run_id"),
                "metrics_source": metrics_source,
            }
    except Exception as exc:  # pragma: no cover - guard defensivo
        out["metric_drift_error"] = str(exc)

    # cobertura de banda por propiedad: una banda honesta contiene el 80%
    # nominal; la desviación por propiedad (CORIN 67%, VILLAS 93% en la
    # corrida de verificación) se alerta como miscalibración
    try:
        if coverage:
            nominal = float(
                (coverage.get("global") or {}).get("nominal", 0.80)
            )
            for row in coverage.get("by_property") or []:
                n = int(row.get("n_obs", 0))
                cov = float(row.get("coverage", 0.0))
                if n >= 30 and abs(cov - nominal) > 0.075:
                    direction = "sub-cubre" if cov < nominal else "sobre-cubre"
                    alerts.append(
                        DriftAlert(
                            metric=f"interval_coverage:{row.get('property')}",
                            severity="warning",
                            message=(
                                f"Cobertura de banda en {row.get('property')}: "
                                f"{cov:.1%} vs {nominal:.0%} nominal ({direction}, "
                                f"n={n}) — banda mal calibrada"
                            ),
                            reference=nominal,
                            current=cov,
                        )
                    )
    except Exception as exc:  # pragma: no cover - guard defensivo
        out["coverage_alert_error"] = str(exc)

    out["n_drift_alerts"] = len(alerts)
    out["drift_alerts"] = _jsonable([a.to_dict() for a in alerts])

    try:
        span_days = int(
            (history["stay_date"].max() - history["stay_date"].min()).days + 1
        )
        # estado real del registro: cuánto tiempo desde el último
        # entrenamiento y cuántos registros NUEVOS desde entonces
        prev_ts = previous.get("timestamp")
        prev_training_end = previous.get("training_end")
        if prev_ts:
            try:
                days_since_last = int(
                    (pd.Timestamp.now(tz="UTC") - pd.Timestamp(prev_ts)).days
                )
            except (ValueError, TypeError, OverflowError):
                days_since_last = 0
        else:
            days_since_last = 0
        if prev_training_end:
            new_records = int(
                (pd.to_datetime(history["stay_date"]) > pd.Timestamp(prev_training_end)).sum()
            )
        else:
            new_records = int(len(history))
        decision = should_retrain(
            frequency=config.retrain_frequency,
            new_records=new_records,
            days_since_last=days_since_last,
            min_history_days=int(config.min_history_days),
            current_history_days=span_days,
        )
        out["retraining_state"] = {
            "days_since_last": days_since_last,
            "new_records": new_records,
            "has_previous_run": bool(prev_ts),
            "previous_run_id": previous.get("run_id"),
        }
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


def _dm_horizon_from_panel(error_panel: pd.DataFrame | None) -> int:
    """Largo REAL de la ventana de fold (pasos) para la corrección HAC de DM.

    El panel de errores trae una fila por fecha evaluada y fold: el número de
    fechas distintas por fold es el horizonte multi-paso de la evaluación.
    Se usa la mediana entre folds (robusta a folds recortados por histórico
    insuficiente). Sin panel evaluable → 30, el default histórico.
    """
    if error_panel is None or error_panel.empty or "stay_date" not in error_panel.columns:
        return 30
    if "fold" in error_panel.columns:
        per_fold = error_panel.groupby("fold")["stay_date"].nunique()
    else:
        per_fold = pd.Series([error_panel["stay_date"].nunique()])
    per_fold = per_fold[per_fold > 0]
    if per_fold.empty:
        return 30
    return int(max(1, round(float(per_fold.median()))))


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
    fold_predictions: pd.DataFrame | None = None,
    error_panel: pd.DataFrame | None = None,
    eval_frame: pd.DataFrame | None = None,
    eval_models: dict[str, str] | None = None,
    backtest_cfg: dict | None = None,
    registry_path: str | Path | None = None,
    run_mode: str = "ad hoc",
) -> E2EResult:
    """Ejecuta la cadena completa y verifica los criterios de aceptación.

    ``fold_metrics``/``error_panel`` inyectan la evaluación REAL (benchmark
    sobre datos reales + panel de errores por fecha para la significancia
    Diebold-Mariano); el CAMPEÓN resultante opera el forecast. Si se
    omiten, el benchmark usa placeholders sintéticos (etiquetados), el
    modelo operativo es ``pickup_curve`` (default documentado) y el
    manifiesto lo marca ``benchmark_source: synthetic_placeholder``.

    ``fold_predictions`` (retención de ``real_fold_metrics``) alimenta los
    residuos del forecast OPERATIVO por (propiedad, banda) para los
    intervalos — la banda refleja la escala de error de cada propiedad
    (spec forecasting). Sin ellas, los intervalos usan el pool del
    portafolio desde el backtest T-90..T-1 (comportamiento previo).

    ``eval_frame`` inyecta la evaluación de métricas y desgloses (columnas
    ``property/stay_date/lead_days/actual/forecast``) cuando la ventana de
    pronóstico todavía no tiene reales —corrida **futura**—: en ese caso las
    métricas provienen del backtest y el manifiesto lo etiqueta como
    ``metrics_source: injected_backtest`` (nunca se inventan reales).
    ``eval_models`` declara el modelo que operó cada propiedad en esa
    evaluación (selección as-of al inicio de la ventana de backtest): el
    manifiesto lo publica como ``eval_model_by_property`` — las métricas
    describen lo que opera, no un modelo distinto.
    """
    config = config or load_config()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # registro entre corridas (estado real del monitoreo): sin ruta →
    # corrida hermética sin estado (tests); con ruta → drift 10.1 contra la
    # corrida anterior, histéresis del campeón y tracking vintage
    previous_run: dict[str, Any] | None = None
    previous_operational: dict[str, Any] | None = None
    registry: dict[str, Any] | None = None
    if registry_path is not None:
        try:
            registry = load_registry(registry_path)
            previous_run = latest_record(registry)
            previous_operational = latest_record(registry, mode="futuro")
        except Exception as exc:  # el registro jamás rompe la corrida
            print(f"[aviso] registro de corridas ilegible ({exc}); corrida sin estado")
            previous_run = None
            previous_operational = None

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

    # ---------- 2b. calendario de feriados/eventos (exógeno, a cualquier lead)
    events_df = load_events(config.events_path)
    holiday_cal = holiday_calendar_from_config(
        config,
        min_year=int(pd.Timestamp(history["stay_date"].min()).year),
        max_year=int((cutoff + pd.Timedelta(days=horizon_days)).year),
    )
    if events_df is not None:
        base_cal = holiday_cal if holiday_cal is not None else HolidayCalendar(
            dates=pd.DatetimeIndex([]), windows=pd.Series(dtype=object),
            countries=(), window_days=int(dict(config.holidays).get("window_days", 4)),
        )
        holiday_cal = calendar_with_events(base_cal, events_df)
        print(
            f"[eventos] {len(events_df)} eventos cargados de '{config.events_path}' "
            f"({int(events_df['property'].notna().sum())} con propiedad asignada)"
        )

    # ---------- 3. features as-of (sobre history, no sobre el futuro)
    df = history.copy()
    # is_holiday: SOLO feriados del calendario público (nunca eventos privados)
    is_holiday_dates = None
    if holiday_cal is not None:
        is_holiday_dates = holiday_cal.holiday_dates
        if is_holiday_dates is None or len(is_holiday_dates) == 0:
            is_holiday_dates = holiday_cal.dates
    df = add_calendar_features(
        df, "stay_date",
        holidays=list(is_holiday_dates) if is_holiday_dates is not None else None,
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
            fold_metrics, error_panel=error_panel, source="real_backtest",
            dm_horizon=_dm_horizon_from_panel(error_panel),
        )
        champion = benchmark.champion
        per_prop_sel = (
            select_per_property(
                fold_metrics,
                global_champion=champion,
                # histéresis: el incumbente de la corrida anterior se
                # retiene salvo ventaja material (empates DM = ruido)
                incumbent=(previous_run or {}).get("model_by_property"),
            )
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
    raw, fitted_models, blend_by_property = build_forecasts(
        model_by_property, history, pick, cutoff, horizon_days,
        curve_params, holiday_cal,
        fold_metrics=fold_metrics,  # evidencia para el blend accuracy-aware
    )

    # ---------- 5.1 corrección de nivel por drift (solo modo futuro)
    # Los errores del forecast PUBLICADO previo miden el desvío de nivel
    # vigente (walk-forward, sin lookahead). En modo verificación NO se
    # aplica: el horizonte con reales no se corrige con información del
    # propio horizonte (honestidad de la evaluación).
    level_cfg = dict((config.monitoring or {}).get("level_correction") or {})
    level_factors: dict[str, float] = {}
    level_report: dict = {"enabled": bool(level_cfg.get("enabled", False)),
                          "run_mode": str(run_mode)}
    if (
        level_cfg.get("enabled")
        and str(run_mode) == "futuro"
        and previous_operational is not None
    ):
        prev_fc = load_published_forecast(
            str(previous_operational.get("run_dir") or "")
        )
        if prev_fc is not None:
            detail_level = vintage_detail(
                prev_fc, dem, through=pd.Timestamp(dem["stay_date"].max())
            )
            level_factors, level_report = level_correction_factors(
                detail_level,
                trigger_ts=float(level_cfg.get("ts_trigger", 4.0)),
                window_days=int(level_cfg.get("window_days", 60)),
                min_obs=int(level_cfg.get("min_obs", 10)),
                shrinkage=int(level_cfg.get("shrinkage", 10)),
                max_rel=float(level_cfg.get("max_rel", 0.20)),
            )
            level_report["published_run_id"] = str(
                previous_operational.get("run_id") or ""
            )
        else:
            level_report["note"] = "sin dataset publicado previo legible"
    elif level_cfg.get("enabled"):
        level_report["note"] = (
            "sin correccion: modo verificacion (el horizonte con reales"
            " no se corrige con informacion del propio horizonte)"
            if str(run_mode) != "futuro"
            else "sin corrida futura previa en el registro"
        )
    raw = apply_level_factors(raw, level_factors)

    # ---------- 5b. corrección de bias por (propiedad, banda de lead)
    # El backtest operativo documenta sobre-pronóstico sistemático a leads
    # largos (+23%/+30% en 60-90+): la corrección se aprende de los residuos
    # del forecast operativo en los folds (shrinkage por tamaño de muestra)
    # y se aplica al punto ANTES de piso/techo. Sin fold_predictions la
    # corrección es vacía (corridas de muestra quedan idénticas).
    # Knob `enabled` (kill-switch) y `fold_recency_decay` (pondera los
    # folds por recencia) — openspec 2026-11-05-bias-correction-mode.
    bias_cfg = dict((config.monitoring or {}).get("bias_correction") or {})
    seasonal_cells = bool(bias_cfg.get("seasonal_cells", False))
    bias_enabled = bool(bias_cfg.get("enabled", True))
    if bias_enabled:
        bias_factors, bias_report = lead_bias_factors(
            fold_predictions, model_by_property, blend_by_property, history,
            min_obs=int(bias_cfg.get("min_obs", 20)),
            shrinkage=int(bias_cfg.get("shrinkage", 30)),
            max_rel=float(bias_cfg.get("max_rel", 0.30)),
            seasonal_cells=seasonal_cells,
            fold_recency_decay=float(bias_cfg.get("fold_recency_decay", 1.0)),
        )
        raw = apply_bias_factors(
            raw, bias_factors,
            fallback_factors=(
                dict(bias_report.get("band_factors") or {}) if seasonal_cells else None
            ),
        )
    else:
        bias_factors, bias_report = {}, {"enabled": False}

    # cold start: propiedades con < 365d de histórico (p. ej. apertura
    # reciente) — forecast provisional, documentado en el manifiesto
    first_stay = history.groupby("property")["stay_date"].min()
    history_days = (pd.Timestamp(cutoff) - first_stay).dt.days + 1
    cold_start_properties = {
        str(p): f"{int(d)}d < 365d de histórico: forecast provisional"
        for p, d in history_days.items()
        if int(d) < 365
    }

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
    # piso de reservas: separa las alertas de piso de las de techo
    floor_alerts = [a for a in cap_alerts if isinstance(a, BookedFloorAlert)]
    capacity_only_alerts = [a for a in cap_alerts if not isinstance(a, BookedFloorAlert)]
    n_floor_applied = int(constrained.get("booked_floor_applied", pd.Series(dtype=bool)).sum())

    # ---------- 7. backtest T-90..T-1 muestreado (residuos por lead band)
    backtest_report, residual_bands, residuals_flat = _sampled_backtest(
        history, pick, model_by_property, cutoff, backtest_cfg, curve_params, holiday_cal
    )

    # ---------- 8. intervalos P10/P50/P90 anclados al punto publicado
    # residuos del forecast OPERATIVO por (propiedad, banda): la banda
    # refleja la escala de error de CADA propiedad (spec forecasting) —
    # el pool del portafolio daba bandas identicas para hoteles de 54 y
    # 408 habitaciones (CORIN sobre-cubria 97-100%, FIESTA sub-cobria
    # 25-37%). Ensamblados de las predicciones que los folds ya
    # computaron, sin re-entrenar nada. Los residuos son los del forecast
    # CORREGIDO (bias_factors): la banda calibra el pipeline completo que
    # se publica — punto corregido + banda — con cuantiles completos del
    # residuo (la cobertura nominal se garantiza sobre la calibración).
    operative_bands = operative_residual_bands(
        fold_predictions, model_by_property, blend_by_property, history,
        bias_factors=bias_factors,
        recency_decay=float((config.monitoring or {}).get("interval_recency_decay", 0.7)),
        seasonal_cells=seasonal_cells,
        fallback_factors=(
            dict(bias_report.get("band_factors") or {}) if seasonal_cells else None
        ),
    )
    residuals_provided = bool(residual_bands) and len(residuals_flat) > 0
    if operative_bands:
        interval = build_intervals(
            with_occ,
            model_name=champion,
            point_col="final_forecast_rn",
            residual_bands=operative_bands,
            property_col="property",
        )
        interval_source = "operative_fold_residuals"
    else:
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
        a["stay_date"] = parse_dates_flexible(a["stay_date"])
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
        eval_src["stay_date"] = parse_dates_flexible(eval_src["stay_date"])
        eval_src["lead_days"] = (
            pd.to_numeric(eval_src["lead_days"], errors="coerce").fillna(30).astype(int)
        )
        metrics = compute_metrics(eval_src["actual"], eval_src["forecast"]).to_dict()
        breakdowns = {
            "lead_band": metrics_by_lead_band(eval_src),
            "property": metrics_by_property(eval_src),
            # grano semanal: lente operativo y honesto para propiedades
            # chicas (el MAPE diario con reales de 4 RN es ruido, no senal)
            "property_weekly": metrics_by_property_weekly(eval_src),
            **metrics_by_seasonality(eval_src),
        }
        # cobertura empírica de la banda P10–P90 sobre la MISMA ventana con
        # reales (corrida de verificación); sin banda sobre la ventana (p. ej.
        # corrida futura: la banda vive en el horizonte, que no tiene reales)
        # queda en None y el accuracy report lo documenta
        coverage = interval_coverage(eval_src)
    else:
        coverage = None

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

    # ---------- 10b. tracking vintage: el forecast publicado ANTERIOR se
    # puntúa contra los reales materializados desde entonces (mide el
    # artefacto que el negocio vio, no una simulación); y la revisión
    # entre corridas (delta del forecast sobre estancias aún futuras)
    vintage_tracking: dict | None = None
    forecast_rev: dict | None = None
    if registry is not None:
        try:
            futuros = [
                r for r in registry.get("records") or []
                if str(r.get("mode")) == "futuro"
            ]
            # revisión entre corridas: contra la última publicación (linaje)
            if previous_operational is not None:
                prev_fc = load_published_forecast(
                    str(previous_operational.get("run_dir") or "")
                )
                if prev_fc is not None:
                    forecast_rev = forecast_revision(forecast_final, prev_fc)
                    if forecast_rev is not None:
                        kpis["forecast_vs_previous"] = float(forecast_rev["delta"])
                        kpis["forecast_vs_previous_pct"] = float(
                            forecast_rev["delta_pct"]
                        )
            # vintage: la publicación MÁS RECIENTE cuya añada ya tiene reales
            # (una futura recién publicada con corte en el último real aún no
            # es graduable — su horizonte entero es futuro)
            seleccion = select_gradable_publication(
                futuros, dem, through=pd.Timestamp(dem["stay_date"].max())
            )
            if seleccion is not None:
                record_annada, vintage_tracking = seleccion
                vintage_tracking["published_run_id"] = str(
                    record_annada.get("run_id") or ""
                )
                # detalle forecast-vs-real por estancia (entregable CSV)
                prev_fc = load_published_forecast(
                    str(record_annada.get("run_dir") or "")
                )
                if prev_fc is not None:
                    detail = vintage_detail(
                        prev_fc,
                        dem,
                        through=pd.Timestamp(dem["stay_date"].max()),
                        published_run_id=str(record_annada.get("run_id") or ""),
                    )
                    if not detail.empty:
                        detail.to_csv(
                            run_dir / "vintage_tracking_detail.csv", index=False
                        )
                        vintage_tracking["detail_csv"] = "vintage_tracking_detail.csv"
                        vintage_tracking["n_detail_rows"] = int(len(detail))
        except Exception as exc:  # el tracking jamás rompe la corrida
            vintage_tracking = {"error": str(exc)}

    # ---------- 11. monitoring (outliers + drift + retraining)
    monitoring = _monitoring_block(
        history, metrics, config,
        cutoff=pd.Timestamp(cutoff),
        metrics_source=metrics_source,
        previous=previous_run,
        coverage=coverage,
    )

    # expediente del forecast publicado: todas las añadas graduadas
    # (deduplicadas, graduación más reciente por añada) desde el registro
    track_record = build_track_record(registry.get("records") or []) if registry else []
    track_summary = {
        "n_annadas_graduadas": len(track_record),
        "annadas": [
            {
                "cutoff": e.get("cutoff"),
                "window": e.get("window"),
                "n_obs": e.get("n_obs"),
                "wmape": e.get("wmape"),
            }
            for e in track_record
        ],
    }

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
            "per_property_blend": _jsonable(blend_by_property) if blend_by_property else {},
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
            "n_booked_floor_alerts": int(
                sum(1 for a in cap_alerts if isinstance(a, BookedFloorAlert) and a.shortfall > 0)
            ),
            "n_booked_over_capacity": int(
                sum(1 for a in cap_alerts if isinstance(a, BookedFloorAlert) and np.isnan(a.raw_forecast))
            ),
            "n_booked_floor_applied": n_floor_applied,
            "kpis": _jsonable(kpis),
            "sample_data": sample_data,
            "benchmark_source": benchmark_source,
            "interval_source": interval_source,
            "metrics_source": metrics_source,
            # modelos que operaron la EVALUACIÓN inyectada (selección as-of al
            # inicio de la ventana de backtest): las métricas publicadas
            # describen estos modelos, no otros
            "eval_model_by_property": (
                _jsonable(eval_models) if eval_models else None
            ),
            "forecast_window": {
                "start": str((cutoff + pd.Timedelta(days=1)).date()),
                "end": str((cutoff + pd.Timedelta(days=horizon_days)).date()),
            },
            "capacity_usage": "horizon_cap_only",
            "excluded_properties": _jsonable(excluded_props_map),
            "cold_start_properties": _jsonable(cold_start_properties),
            "bias_correction": _jsonable(bias_report),
            "level_correction": _jsonable(level_report),
            "vintage_tracking": _jsonable(vintage_tracking),
            "forecast_revision": _jsonable(forecast_rev),
            "track_record": _jsonable(track_summary),
            "previous_run": (
                {
                    "run_id": previous_run.get("run_id"),
                    "mode": previous_run.get("mode"),
                    "timestamp": previous_run.get("timestamp"),
                }
                if previous_run
                else None
            ),
            "holiday_calendar": _jsonable(holiday_cal.describe()) if holiday_cal is not None else None,
            "events": (
                {
                    "source": str(config.events_path),
                    "n_events": int(len(events_df)),
                    "n_scoped": int(events_df["property"].notna().sum()),
                }
                if events_df is not None
                else {"source": None}
            ),
            "champion_reason": benchmark.selection.reason,
            "champion_significance": _jsonable(benchmark.selection.significance),
            "property_fallback": property_fallback,
            "evaluation_coverage": _jsonable(evaluation_coverage),
            "monitoring": monitoring,
        },
    )
    manifest_path = manifest.write(run_dir)

    # ---------- 12b. registro de estado entre corridas
    # La SIGUIENTE corrida monitorea contra esta: drift de métricas (10.1),
    # re-entrenamiento con días/nuevos-reales reales (10.4), histéresis del
    # campeón por propiedad y tracking vintage del forecast publicado.
    # El registro es append-only con escritura atómica y jamás rompe.
    if registry_path is not None:
        try:
            coverage_compact = (
                {
                    "global": coverage.get("global"),
                    "by_property": coverage.get("by_property"),
                }
                if coverage
                else {}
            )
            append_record(
                registry_path,
                RunRecord(
                    run_id=run_dir.name,
                    run_dir=str(run_dir),
                    mode=str(run_mode),
                    timestamp=run_ts,
                    cutoff=str(pd.Timestamp(cutoff).date()),
                    training_end=str(pd.Timestamp(cutoff).date()),
                    metrics=_jsonable(metrics) if metrics else {},
                    metrics_source=str(metrics_source),
                    champion=str(champion),
                    model_by_property=dict(model_by_property),
                    interval_coverage=_jsonable(coverage_compact),
                    vintage_tracking=_jsonable(vintage_tracking) or {},
                ),
            )
        except Exception as exc:  # pragma: no cover - guard defensivo
            print(f"[aviso] no se pudo actualizar el registro de corridas: {exc}")

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
            "by_property_weekly": _jsonable(
                breakdowns.get("property_weekly", pd.DataFrame()).to_dict("records")
            ),
            "evaluation_coverage": _jsonable(evaluation_coverage),
            # cobertura empírica P10–P90: una banda honesta contiene el 80%
            # nominal de los reales; el desglose por propiedad/banda expone
            # las celdas mal calibradas (p. ej. P90 estrecho en la propiedad
            # más grande). Sin banda sobre la ventana evaluada se documenta.
            "interval_coverage": (
                _jsonable(coverage)
                if coverage is not None
                else {
                    "note": (
                        "banda P10-P90 no disponible sobre la ventana evaluada "
                        "(la banda se publica sobre el horizonte de pronóstico, "
                        "que aún no tiene reales)"
                    )
                }
            ),
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
            interval_coverage=coverage,
            eval_models=eval_models,
            excluded_props_map=excluded_props_map,
            curve_params=curve_params,
            holiday_calendar_md=(
                f"{holiday_cal.describe()['n_holiday_dates']} fechas "
                f"{'+'.join(holiday_cal.countries)} (ventana ±{holiday_cal.window_days}d)"
                if holiday_cal is not None
                else None
            ),
            cold_start_properties=cold_start_properties,
            bias_report=bias_report,
            vintage_tracking=vintage_tracking,
            forecast_revision=forecast_rev,
        ),
    )

    # ---------- checks
    # con piso de booked: final >= booked cuando booked existe (spec forecasting)
    booked_vals = pd.to_numeric(fc_table.get("current_booked_rn"), errors="coerce")
    floor_ok = True
    if booked_vals is not None:
        mask = booked_vals.notna() & (booked_vals <= fc_table["capacity"] + 1e-9)
        floor_ok = bool(
            (fc_table.loc[mask, "final_forecast_rn"] >= booked_vals[mask] - 1e-9).all()
        )
    capacity_ok = bool(
        (fc_table["final_forecast_rn"] <= fc_table["capacity"] + 1e-9).all()
        and not forecast_problems
        and floor_ok
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
            "n_booked_floor_applied": n_floor_applied,
            "n_booked_floor_alerts": int(
                sum(1 for a in cap_alerts if isinstance(a, BookedFloorAlert) and a.shortfall > 0)
            ),
            "n_booked_over_capacity": int(
                sum(1 for a in cap_alerts if isinstance(a, BookedFloorAlert) and np.isnan(a.raw_forecast))
            ),
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
        actual_note: str | None = None
        if eval_src is not None and "actual" in eval_src.columns:
            n_reales_eval = int(
                pd.to_numeric(eval_src["actual"], errors="coerce").notna().sum()
            )
            merged_actual = eval_src[["property", "stay_date", "actual"]].drop_duplicates(
                subset=["property", "stay_date"]
            )
            joined = forecast_final.reset_index(drop=True).merge(
                merged_actual, on=["property", "stay_date"], how="left"
            )
            if joined["actual"].notna().any():
                actual_series = joined["actual"]
            elif n_reales_eval:
                # reales EXISTEN pero la unión no encontro coincidencias:
                # la comparativa declara el motivo — nunca un hueco
                # silencioso (openspec 2026-11-09-report-legibility)
                actual_note = (
                    "comparativa real-vs-forecast no disponible: la union de "
                    "reales con el horizonte publicado no encontro coincidencias "
                    "(revisar alineacion de claves property/stay_date)"
                )
        # corrida futura: la banda vive sobre el horizonte (sin reales) — la
        # fila de cobertura del panel se presenta como N/D con el motivo,
        # nunca se omite silenciosamente (spec reporting: auto-explicado)
        coverage_for_report: dict | None = coverage
        if coverage_for_report is None and eval_frame is not None:
            coverage_for_report = {
                "note": (
                    "banda P10-P90 no disponible sobre la ventana evaluada "
                    "(corrida futura: la banda se publica sobre el horizonte "
                    "de pronóstico, que aún no tiene reales)"
                )
            }
        generate_visual_report(
            run_dir,
            forecast_final,
            demand=dem,
            benchmark=benchmark.table,
            metrics=metrics,
            kpis=kpis,
            actual=actual_series,
            actual_note=actual_note,
            property_breakdown=breakdowns.get("property"),
            property_weekly_breakdown=breakdowns.get("property_weekly"),
            model_by_property=model_by_property,
            backtest_report=backtest_report,
            champion=champion,
            # panel de interpretación y validación (auto-explicado)
            significance=benchmark.selection.significance,
            interval_coverage=coverage_for_report,
            cold_start=cold_start_properties,
            drift_alerts=monitoring.get("drift_alerts"),
            monitoring_state=monitoring.get("retraining_state"),
            bias_correction=bias_report,
            vintage_tracking=vintage_tracking,
            forecast_revision=forecast_rev,
            lead_band_breakdown=(
                breakdowns.get("lead_band").to_dict("records")
                if breakdowns.get("lead_band") is not None
                and not breakdowns["lead_band"].empty
                else None
            ),
            benchmark_bands=dict(config.monitoring or {}).get("benchmark_bands"),
            tracking_signal_alert=float(
                dict(config.monitoring or {}).get("tracking_signal_alert", 4.0)
            ),
            track_record=track_record,
            run_id=run_dir.name,
            run_mode=(
                str(run_mode) if str(run_mode) in ("real", "futuro") else None
            ),
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
    interval_coverage: dict | None = None,
    eval_models: dict[str, str] | None = None,
    excluded_props_map: dict[str, str] | None = None,
    curve_params: dict | None = None,
    holiday_calendar_md: str | None = None,
    cold_start_properties: dict[str, str] | None = None,
    bias_report: dict | None = None,
    vintage_tracking: dict | None = None,
    forecast_revision: dict | None = None,
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
        "corte (la ventana de pronóstico es futura y aún no tiene reales). La "
        "selección del campeón se repite **as-of** al inicio de esa ventana y el "
        "campeón resultante la opera: las métricas miden el pipeline completo "
        "(selección + forecast) sin ventaja retrospectiva. El manifiesto lo marca "
        "como `metrics_source: injected_backtest` y declara los modelos que "
        "operaron la evaluación (`eval_model_by_property`).\n"
        if metrics_source == "injected_backtest"
        else "- **Origen de las métricas**: reales observados dentro de la ventana del horizonte.\n"
    )
    per_prop = model_by_property or {}
    per_prop_md = (
        "\n".join(f"  - {p}: `{m}`" for p, m in sorted(per_prop.items())) or "  - (todas con el campeón global)"
    )
    if eval_models:
        eval_md = (
            "\n".join(f"  - {p}: `{m}`" for p, m in sorted(eval_models.items()))
            or "  - (campeón global)"
        )
        eval_block = (
            "### Modelos que operaron la evaluación (selección as-of al inicio de la "
            "ventana de backtest)\n"
            f"{eval_md}\n"
        )
    else:
        eval_block = ""
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
        "residuos del forecast operativo por (propiedad × banda de lead) de los "
        "folds del benchmark, corregidos por bias; cuantiles COMPLETOS del residuo "
        "(calibración conformal: la banda contiene el 80% nominal de la muestra "
        "de calibración por construcción)"
        if interval_source == "operative_fold_residuals"
        else (
            "residuos del backtest T-90..T-1 por banda de lead (cuantiles completos)"
            if interval_source == "real_backtest"
            else "residuos sintéticos de reemplazo (etiquetados en el manifiesto)"
        )
    )
    if interval_coverage is not None:
        g = interval_coverage.get("global", {})
        cov_pct = float(g.get("coverage", float("nan"))) * 100
        n_cov = int(g.get("n_obs", 0))
        coverage_md = (
            f"- Cobertura empírica en la ventana evaluada: {cov_pct:.1f}% "
            f"(nominal 80%, n={n_cov}) — desglose por propiedad, banda de lead "
            "y celda (propiedad × banda) en `interval_coverage` del accuracy "
            "report; la desviación por propiedad se alerta en el monitoreo.\n"
        )
    else:
        coverage_md = (
            "- Cobertura empírica: no medible en esta corrida (la banda vive sobre "
            "el horizonte de pronóstico, que aún no tiene reales).\n"
        )
    cold_start_md = (
        "- **Cold start (provisional)**: "
        + (
            "; ".join(f"{p} ({why})" for p, why in sorted(cold_start_properties.items()))
            + " — su forecast es provisional y se excluye de decisiones de "
            "planeación hasta acumular ≥ 365d de histórico.\n"
            if cold_start_properties
            else "ninguna propiedad con < 365d de histórico.\n"
        )
    )
    bias_md = (
        "- **Corrección de bias por (propiedad, banda de lead)**: aprendida de los "
        "residuos del forecast operativo en los folds (sobre-pronóstico documentado "
        "de +23%/+30% a leads 60–90+), con shrinkage por tamaño de muestra y clip "
        "±30%; se aplica al punto ANTES del piso de reservas y el techo de "
        "capacidad. Detalle por celda en el manifiesto (`bias_correction`): "
        + (
            f"{int(bias_report.get('n_corrected', 0))} de "
            f"{int(bias_report.get('n_cells', 0))} celdas corregidas."
            if bias_report
            else "sin folds de evidencia (sin corrección)."
        )
        + "\n"
    )
    if vintage_tracking:
        vg = vintage_tracking.get("global") or {}
        vintage_md = (
            "- **Tracking vintage**: el forecast publicado por la corrida anterior "
            "se puntuó contra los reales materializados "
            + (
                f"(WMAPE {float(vg.get('wmape', float('nan'))):.4f} sobre "
                f"{int(vintage_tracking.get('n_obs', 0))} obs; detalle por banda y "
                "cobertura de banda por vintage en `vintage_tracking` del manifiesto"
                + (
                    f"; detalle por estancia — publicado vs real, error y banda — en "
                    f"`{vintage_tracking.get('detail_csv')}` "
                    f"({int(vintage_tracking.get('n_detail_rows', 0))} filas)"
                    if vintage_tracking.get("detail_csv")
                    else ""
                )
                + ").\n"
                if vg
                else "(sin estancias materializadas aún).\n"
            )
        )
    else:
        vintage_md = (
            "- **Tracking vintage**: sin forecast operativo (modo `futuro`) previo "
            "registrado, o sin estancias de ese forecast materializadas aún — se "
            "activa en cuanto exista una corrida futura previa con reales.\n"
        )
    if forecast_revision:
        revision_md = (
            f"- **Revisión vs corrida anterior**: {float(forecast_revision['delta']):+.0f} RN "
            f"({float(forecast_revision['delta_pct']):+.1%}) sobre "
            f"{int(forecast_revision['n_stays'])} estancias comunes aún futuras "
            "(`forecast_revision` en el manifiesto).\n"
        )
    else:
        revision_md = ""
    monitoring_md = (
        "- **Monitoreo con estado**: drift de demanda por propiedad contra la misma "
        "ventana YoY (PSI — la referencia 'primera mitad del histórico' disparaba "
        "en cada corrida por el crecimiento del portafolio), drift de métricas "
        "contra la corrida anterior registrada y política de re-entrenamiento con "
        "días/nuevos-reales reales del registro. La selección por propiedad "
        "retiene al incumbente salvo ventaja material (histéresis anti-churn).\n"
    )
    return f"""# Documentación técnica — Room Nights Forecasting Engine

{disclaimer}

## Modelo operativo
- **Campeón**: `{champion}` — seleccionado por el benchmark ({bench_note}) y
  **operado** en esta corrida (el forecast lo genera el campeón).
- **Curva de pickup**: {curve_md}
- **Selección por propiedad**:
{per_prop_md}
{eval_block}- **Propiedades cerradas (excluidas)**:
{excluded_md}
- La significancia de la selección (Diebold-Mariano) queda en el manifiesto
  (`champion_significance`); un empate estadístico se desempata por parsimonia.
- El override por propiedad exige un margen de materialidad sobre el campeón
  global (diferencias pequeñas entre 4 folds son ruido, no evidencia).

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
{coverage_md}
## Monitoreo y operación
{monitoring_md}{bias_md}{vintage_md}{revision_md}{cold_start_md}
## Limitaciones del modelo
1. Sin datos externos: clima, competencia y eventos propietarios fuera de alcance.
2. ETS/SARIMAX se entrenan sobre series posicionales: con huecos severos el paso
   temporal se aproxima; los candidatos ML usan features calendario exactas.
3. La banda de un lead con pocos residuos observados hereda la escalera de pools
   ((propiedad, banda) → propiedad → banda → global, documentada con
   `min_residuals`).
4. Cold start: propiedades con < 60 días de histórico usan guard de nivel
   reciente; con < 365d se publican como **provisionales** (ver arriba).
5. Revenue (`RN × ADR`) queda fuera de esta versión.
6. Granularidad `Property × Stay Date`; dimensiones más finas solo si las fuentes
   las soportan de forma consistente.
7. La corrección de bias y la banda se calibran con la evidencia de folds
   disponible; un cambio de régimen posterior al corte solo lo detecta el
   monitoreo (drift YoY por propiedad y tracking vintage), no la calibración.
"""
