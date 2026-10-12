"""Corrida end-to-end REAL sobre las tres fuentes en ``data/`` (task 10.5).

Uso::

    python -m src.forecasting.run_real              # verificación (backtest con reales)
    python -m src.forecasting.run_real --futuro     # pronóstico futuro (default: hasta 31-dic)
    python -m src.forecasting.run_real --futuro --hasta 2026-12-31

Qué hace, sin tocar código:

1. Lee ``data/capacity_daily.csv``, ``data/demand_daily.csv`` y
   ``data/pickup_by_lead.csv`` (rutas del ``config.yaml``).
2. Benchmark REAL por propiedad y portafolio: folds trimestrales (cortes a
   ``spacing·(n_folds..1)`` días del corte) × ventana del horizonte operativo
   completo (``benchmark.fold_horizon_days``, default 90), 7 candidatos. Con
   spacing = ventana, las 4 ventanas tesian el año anterior al corte —
   cada estación evaluada una vez — y NO solapan la ventana de métricas
   publicada. Se emite además un panel de errores por fecha para la prueba
   de significancia Diebold-Mariano del campeón.
3. Modo verificación (default): corte ``max_stay - horizonte`` para que el
   horizonte quede cubierto por reales observados (verificación honesta de
   las métricas).
4. Modo ``--futuro``: corte = último real y horizonte hasta ``--hasta``.
   La ventana futura NO tiene reales, así que métricas y desgloses provienen
   de un backtest real sobre la ventana ANTERIOR al corte
   (``metrics_source: injected_backtest`` en el manifiesto); jamás se
   inventan reales futuros.
5. El CAMPEÓN del benchmark opera el forecast: el runner lo despacha por
   propiedad (selección por propiedad cuando hay evidencia suficiente) y
   ejecuta el backtest T-90..T-1 muestreado que alimenta los residuos por
   banda de lead de P10/P50/P90 y el entregable ``backtesting_report.json``.
6. Verifica los criterios de aceptación y sale con código ≠ 0 si alguno falla.

Salida: ``outputs/runs/real-<fecha>/`` o ``outputs/runs/futuro-<fecha>/``
con los 10 entregables + manifiesto.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data.schema import SourceKind, detect_schema
from src.evaluation.metrics import compute_metrics
from src.forecasting.core import (
    CANDIDATE_MODELS,
    PORTFOLIO,
    _apply_aliases,
    _booked_requests,
    _normalize_capacity,
    _normalize_demand,
    _normalize_pickup,
    _remaining_pickup_curves,
    fit_candidate,
    predict_candidate,
)
from src.forecasting.runner import run_end_to_end
from src.features.events import load_events
from src.features.holidays import (
    HolidayCalendar,
    calendar_with_events,
    holiday_calendar_from_config,
)
from src.models.baselines import PickupCurveModel

# Ventana de cada fold = horizonte OPERATIVO completo (config
# ``benchmark.fold_horizon_days``). Con spacing trimestral y 4 folds, las 4
# ventanas tesian el año anterior al corte: cada estación evaluada una vez.
# (Antes 30d: la selección era ciega al régimen largo leads 30–90.)
FOLD_HORIZON = 90  # ventana de cada fold (días)
FOLD_SPACING = 90  # separación trimestral entre folds
ML_MIN_TRAIN_OBS = 60


# --------------------------------------------------------------------------
# carga de las tres fuentes reales (crudas; el runner aplica alias/esquema)
# --------------------------------------------------------------------------

def load_real_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lee los CSVs reales y devuelve (demand, capacity, pickup, demand_aliased).

    Las propiedades cerradas (``data.excluded_properties``) se filtran tras
    resolver alias: no cruzan al benchmark ni a la corrida end-to-end.
    """
    cfg = load_config()
    paths = cfg.source_paths
    demand = pd.read_csv(paths["demand"])
    capacity = pd.read_csv(paths["capacity"])
    pickup = pd.read_csv(paths["pickup"])

    dem_schema = detect_schema(demand, SourceKind.DEMAND, cfg.column_hints)
    cap_schema = detect_schema(capacity, SourceKind.CAPACITY, cfg.column_hints)
    pk_schema = detect_schema(pickup, SourceKind.PICKUP, cfg.column_hints)

    demand = _apply_aliases(demand, dem_schema, cfg.property_aliases)
    capacity = _apply_aliases(capacity, cap_schema, cfg.property_aliases)
    pickup = _apply_aliases(pickup, pk_schema, cfg.property_aliases)

    excluded = [str(p) for p in cfg.excluded_properties]
    if excluded:
        demand = demand.loc[~demand[dem_schema.property_col].astype(str).isin(excluded)]
        capacity = capacity.loc[~capacity[cap_schema.property_col].astype(str).isin(excluded)]
        pickup = pickup.loc[~pickup[pk_schema.property_col].astype(str).isin(excluded)]
        print(f"      propiedades cerradas excluidas: {excluded}")

    dem = _normalize_demand(demand, dem_schema)
    cap = _normalize_capacity(capacity, cap_schema)
    pick = _normalize_pickup(pickup, pk_schema)
    return dem, cap, pick, demand


# --------------------------------------------------------------------------
# benchmark real: folds trimestrales × propiedades + portafolio bottom-up
# --------------------------------------------------------------------------


def fold_cutoffs_for(
    cutoff: pd.Timestamp, n_folds: int = 4, spacing: int = FOLD_SPACING
) -> list[pd.Timestamp]:
    """Cortes de los folds de selección: spacing días entre ventanas.

    Con ``spacing == horizon`` las ventanas ``[c+1, c+horizon]`` son
    contiguas y tesian exactamente los ``spacing·n_folds`` días previos al
    corte; el último fold termina en ``cutoff`` y por lo tanto NO solapa la
    ventana de reporte ``(cutoff, cutoff+horizonte]``.
    """
    return [
        pd.Timestamp(cutoff) - pd.Timedelta(days=spacing * (n_folds - i))
        for i in range(n_folds)
    ]


def real_fold_metrics(
    dem: pd.DataFrame, pick: pd.DataFrame, cutoff: pd.Timestamp,
    horizon: int = FOLD_HORIZON, n_folds: int = 4,
    curve_params: dict | None = None,
    holiday_calendar=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas out-of-sample REALES de los 7 candidatos, por propiedad.

    Cada fold entrena con la historia ≤ corte del fold y evalúa los
    ``horizon`` días siguientes contra reales, usando el MISMO pipeline
    fit/predict de producción (``core.fit_candidate``/``predict_candidate``).
    Folds trimestrales: cortes a ``spacing·(n_folds..1)`` días del corte →
    cubren estaciones distintas y no solapan la ventana de reporte
    ``(cutoff, cutoff+horizonte]``.

    Returns
    -------
    (fold_metrics, error_panel):
        ``fold_metrics``: model/fold/property/mape/wmape/mae/rmse/bias (una
        fila por candidato × propiedad × fold; ``PORTFOLIO`` = suma
        bottom-up de las propiedades evaluadas en el fold).
        ``error_panel``: errores por fecha al grano portafolio (base de la
        prueba Diebold-Mariano campeón vs subcampeón).
    """
    props = sorted(dem["property"].dropna().unique())
    fold_cutoffs = fold_cutoffs_for(cutoff, n_folds)

    rows: list[dict] = []
    errors: list[dict] = []
    for fi, c in enumerate(fold_cutoffs):
        dates = pd.date_range(c + pd.Timedelta(days=1), periods=horizon, freq="D")
        per_prop: dict[str, dict[str, pd.Series]] = {}
        actual_by_prop: dict[str, pd.Series] = {}
        viable_by_model: dict[str, set[str]] = {m: set() for m in CANDIDATE_MODELS}
        for p in props:
            dem_p = dem.loc[dem["property"] == p]
            pick_p = pick.loc[pick["property"] == p]
            # grilla diaria COMPLETA de la propiedad: el train se recorta as-of
            # en el fit (fit_candidate filtra ≤ corte); los reales de la
            # ventana del fold viven DESPUÉS del corte del fold
            full = dem_p.groupby("stay_date")["room_nights"].mean().sort_index()
            if full.empty:
                continue
            grid = pd.date_range(full.index.min(), full.index.max(), freq="D")
            vals = full.reindex(grid).to_numpy(dtype=float)
            series_daily = pd.Series(vals, index=grid)
            actual_win = series_daily.reindex(dates).dropna()
            train_obs = series_daily.loc[series_daily.index <= c].dropna()
            if len(train_obs) < ML_MIN_TRAIN_OBS or actual_win.empty:
                continue
            preds: dict[str, pd.Series] = {}
            for model in CANDIDATE_MODELS:
                fitted = fit_candidate(
                    model, dem_p, pick_p, c, horizon, curve_params, holiday_calendar
                )
                pred = predict_candidate(fitted, pick_p, dates)
                preds[model] = pred.set_index("stay_date")["raw_forecast_rn"]
            per_prop[p] = preds
            actual_by_prop[p] = actual_win

        if not per_prop:
            continue
        # fechas donde TODAS las propiedades evaluadas tienen actual → portafolio honesto
        common = None
        for p in per_prop:
            idx = actual_by_prop[p].index
            common = idx if common is None else common.intersection(idx)
        evaluated = sorted(per_prop)

        for model in CANDIDATE_MODELS:
            for p in evaluated:
                met = compute_metrics(
                    actual_by_prop[p].to_numpy(),
                    per_prop[p][model].reindex(actual_by_prop[p].index).to_numpy(),
                )
                rows.append({"model": model, "fold": fi, "property": p, **met.to_dict()})
            if common is not None and len(common) > 0:
                port_pred = pd.concat(
                    [per_prop[p][model].reindex(common) for p in evaluated], axis=1
                ).sum(axis=1)
                port_actual = pd.concat(
                    [actual_by_prop[p].reindex(common) for p in evaluated], axis=1
                ).sum(axis=1)
                met = compute_metrics(port_actual.to_numpy(), port_pred.to_numpy())
                rows.append({"model": model, "fold": fi, "property": PORTFOLIO, **met.to_dict()})
                for d, a, f in zip(common, port_actual.to_numpy(), port_pred.to_numpy()):
                    errors.append(
                        {"model": model, "fold": fi, "stay_date": d, "actual": a, "forecast": f}
                    )
    return pd.DataFrame(rows), pd.DataFrame(errors)


# --------------------------------------------------------------------------
# backtest por propiedad (mismo pipeline del runner): eval de corrida futura
# --------------------------------------------------------------------------

def per_property_backtest(
    dem: pd.DataFrame, pick: pd.DataFrame, cutoff: pd.Timestamp, horizon: int = 30,
    curve_params: dict | None = None,
) -> pd.DataFrame:
    """Backtest por propiedad vía el MISMO pipeline del runner.

    Usa ``_remaining_pickup_curves`` + ``_booked_requests`` con corte
    ``cutoff - horizon``, de modo que la ventana evaluada está completamente
    cubierta por reales observados. Devuelve ``property, stay_date, lead_days,
    forecast, actual``: en la corrida futura es la base de las métricas y
    desgloses (no hay reales en el horizonte futuro; nunca se inventan).
    """
    c = cutoff - pd.Timedelta(days=horizon)
    hist = dem.loc[dem["stay_date"] <= c]
    pkf = pick.loc[pick["snapshot_date"] <= c]
    curves = _remaining_pickup_curves(pkf, hist, c)
    curves = curves.loc[curves["lead_days"] <= horizon]
    if curves.empty:
        return pd.DataFrame(columns=["property", "stay_date", "lead_days", "forecast", "actual"])
    model = PickupCurveModel(**(curve_params or {})).fit(curves)

    dates = pd.date_range(c + pd.Timedelta(days=1), periods=horizon, freq="D")
    props = sorted(dem["property"].dropna().unique())
    requests = _booked_requests(pkf, props, dates, c)
    no_booked = requests["current_booked_rn"].isna()
    requests["current_booked_rn"] = requests["current_booked_rn"].fillna(0.0)
    fc = model.predict(requests)
    if no_booked.any():
        level = hist.groupby("property")["room_nights"].mean()
        fc.loc[no_booked, "forecast_rn"] = (
            fc.loc[no_booked, "property"].map(level).fillna(fc.loc[no_booked, "forecast_rn"])
        )

    actual = dem[["property", "stay_date", "room_nights"]].rename(
        columns={"room_nights": "actual"}
    )
    merged = fc.merge(actual, on=["property", "stay_date"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["property", "stay_date", "lead_days", "forecast", "actual"])
    out = merged.rename(columns={"forecast_rn": "forecast"})
    out["lead_days"] = pd.to_numeric(out["lead_days"], errors="coerce").fillna(horizon).astype(int)
    return out[["property", "stay_date", "lead_days", "forecast", "actual"]].reset_index(
        drop=True
    )


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.forecasting.run_real",
        description="Corrida real end-to-end sobre data/*.csv",
    )
    parser.add_argument(
        "--futuro", "--future", action="store_true",
        help=("pronostica desde el último real hasta --hasta (default: fin de año); "
              "métricas y banda provienen de un backtest real previo, porque la "
              "ventana futura aún no tiene reales"),
    )
    parser.add_argument(
        "--hasta", "--until", default=None, metavar="YYYY-MM-DD",
        help="fecha final inclusive del horizonte futuro (default: 31-dic del último año con reales)",
    )
    args = parser.parse_args(argv)

    t0 = time.time()
    cfg = load_config()
    print("[1/5] Leyendo fuentes reales...")
    dem, cap, pick, demand_raw = load_real_sources()
    max_stay = pd.Timestamp(dem["stay_date"].max())

    if args.futuro:
        hasta = (
            pd.Timestamp(args.hasta) if args.hasta
            else pd.Timestamp(max_stay.year, 12, 31)
        )
        horizon = int((hasta - max_stay).days)
        if horizon <= 0:
            print(
                f"ERROR: el último real ({max_stay.date()}) ya alcanza {hasta.date()}; "
                "use --hasta con una fecha posterior al último real.",
                file=sys.stderr,
            )
            return 2
        cutoff = max_stay
        cfg = replace(cfg, forecast_horizon=horizon)
        modo = f"FUTURO {cutoff.date() + pd.Timedelta(days=1)} -> {hasta.date()}"
        run_kind = "futuro"
    else:
        horizon = int(cfg.forecast_horizon)
        cutoff = max_stay - pd.Timedelta(days=horizon)
        modo = f"VERIFICACIÓN (corte {horizon}d antes del último real)"
        run_kind = "real"

    print(
        f"      demand={len(dem):,} filas | capacity={len(cap):,} | "
        f"pickup={len(pick):,} | corte={cutoff.date()} | horizonte={horizon}d"
    )
    print(f"      modo: {modo}")

    # ventanas de fold del benchmark (config ``benchmark``; sin tocar código)
    bench_cfg = dict(cfg.benchmark or {})
    fold_horizon = int(bench_cfg.get("fold_horizon_days", FOLD_HORIZON))
    n_folds = int(bench_cfg.get("n_folds", 4))
    print(
        f"      folds de selección: {n_folds} × {fold_horizon}d "
        f"(tesing {n_folds * FOLD_SPACING}d previos al corte)"
    )

    print(f"[2/5] Benchmark real (folds trimestrales × {fold_horizon}d, 7 candidatos, por propiedad)...")
    curve_params = dict(cfg.pickup_curve)
    holiday_cal = holiday_calendar_from_config(
        cfg,
        min_year=int(pd.Timestamp(dem["stay_date"].min()).year),
        max_year=int(pd.Timestamp(dem["stay_date"].max()).year) + 1,
    )
    events_df = load_events(cfg.events_path)
    if events_df is not None:
        base_cal = holiday_cal if holiday_cal is not None else HolidayCalendar(
            dates=pd.DatetimeIndex([]), windows=pd.Series(dtype=object),
            countries=(), window_days=int(dict(cfg.holidays).get("window_days", 4)),
        )
        holiday_cal = calendar_with_events(base_cal, events_df)
    if holiday_cal is not None:
        desc = holiday_cal.describe()
        print(f"      feriados/eventos: {desc['n_dates']} fechas "
              f"({desc['n_holiday_dates']} feriados) {desc['countries']} "
              f"(ventana ±{holiday_cal.window_days}d)")
    if events_df is not None:
        print(f"      eventos privados: {len(events_df)} cargados de '{cfg.events_path}'")
    folds, error_panel = real_fold_metrics(
        dem, pick, cutoff, horizon=fold_horizon, n_folds=n_folds,
        curve_params=curve_params, holiday_calendar=holiday_cal,
    )
    if folds.empty:
        print("ERROR: el benchmark no produjo evaluaciones (¿histórico insuficiente?).", file=sys.stderr)
        return 1
    port = folds.loc[folds["property"] == PORTFOLIO]
    print(f"      {len(folds)} evaluaciones. Media por modelo (portafolio):")
    print(
        port.groupby("model")[["mape", "wmape"]]
        .mean().sort_values("wmape").round(3).to_string()
    )

    eval_frame: pd.DataFrame | None = None
    if args.futuro:
        print(f"[3/5] Backtest real T-{horizon}d (métricas y desgloses)...")
        eval_bt = per_property_backtest(dem, pick, cutoff, horizon=horizon, curve_params=curve_params)
        if eval_bt.empty:
            print("ERROR: el backtest previo al corte no produjo observaciones reales.", file=sys.stderr)
            return 1
        eval_frame = eval_bt
        print(
            f"      ventana {eval_bt['stay_date'].min().date()} -> "
            f"{eval_bt['stay_date'].max().date()} | n={len(eval_bt)} obs"
        )
    else:
        print("[3/5] Evaluación contra reales de la ventana del horizonte...")

    print("[4/5] Corrida end-to-end completa (campeón opera + backtest T-90..T-1)...")
    result = run_end_to_end(
        demand=demand_raw,
        capacity=cap,
        pickup=pick,
        run_dir=Path("outputs") / "runs" / f"{run_kind}-{pd.Timestamp.now():%Y%m%d-%H%M}",
        config=cfg,
        cutoff=cutoff,
        actual=None if args.futuro else dem,
        sample_data=False,
        fold_metrics=folds,
        error_panel=error_panel,
        eval_frame=eval_frame,
    )

    checks = result.checks
    print(f"\nEntregables: {result.deliverables}")
    print(f"Manifiesto:  {result.manifest_path}")
    print(f"Campeón:     {result.champion}")
    print(f"Alertas capacidad: {len(result.capacity_alerts)}")
    print(
        f"Métricas reales: MAPE={result.metrics.get('mape', float('nan')):.2f}% "
        f"WMAPE={result.metrics.get('wmape', float('nan')):.4f}"
    )
    print(f"Criterios de aceptación: {'TODO OK' if checks.all_passed else checks.failures()}")
    print(f"Detalles: {checks.details}")
    print(f"Tiempo total: {time.time() - t0:.0f}s")
    return 0 if checks.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
