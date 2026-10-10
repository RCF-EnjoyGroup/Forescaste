"""Tests de la remediación model-validation.

Cada test verifica un hallazgo del diagnóstico de especialista:
- 1.3 rollout ML sin off-by-one (rampa lineal, no forecast plano)
- 2.1 folds trimestrales sin solape con la ventana de reporte + grano por
  propiedad
- 2.3 significancia Diebold-Mariano (empate → parsimonia)
- 3.1/3.2/3.3 el campeón del benchmark OPERA el forecast, por propiedad, y
  es coherente en todos los entregables
- 4.1 gate de calidad con dientes en la ruta productiva
- 4.2 capacidad futura dentro del perímetro de leakage
- 5.x bandas ancladas al punto publicado, capadas a capacidad, calibradas
  por lead; backtesting report con métricas reales
- 6.x cobertura de evaluación, KPIs serializables, drivers reales
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import diebold_mariano
from src.forecasting.core import ML_FEATURES, fit_candidate, predict_candidate
from src.forecasting.runner import _explain_forecast, run_end_to_end
from src.forecasting.run_real import FOLD_HORIZON, FOLD_SPACING, real_fold_metrics
from src.monitoring.run_manifest import read_manifest
from src.reporting.benchmark import benchmark_text, run_benchmark
from src.validation.gate import CriticalDataError
from src.validation.leakage import LeakageError, run_leakage_suite
from src.features.asof import LeakageError as AsOfLeakageError


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _demand_2props(n=400, start="2025-01-01", seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    dow = dates.dayofweek
    base = 55 + 12 * (dow >= 5)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    vals = np.clip(base + seasonal + rng.normal(0, 5, n), 0, 100).round()
    return pd.DataFrame(
        {
            "property": ["corin"] * n + ["fiesta"] * n,
            "stay_date": list(dates) + list(dates),
            "rooms_sold": list(vals) + list(vals * 0.5),
        }
    )


def _capacity_2props(n=400, start="2025-01-01", cap=90):
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "propiedad": ["corin"] * n + ["fiesta"] * n,
            "fecha": list(dates) + list(dates),
            "inventory": [cap] * (2 * n),
        }
    )


def _pickup_2props(demand: pd.DataFrame):
    rows = []
    rng = np.random.default_rng(12)
    for prop, g in demand.groupby("property"):
        final = g.set_index("stay_date")["rooms_sold"]
        for stay in g["stay_date"].iloc[::10]:
            for lead in [90, 60, 30, 14, 7, 3, 1]:
                snap = stay - pd.Timedelta(days=lead)
                if snap > pd.Timestamp("2025-06-30"):
                    continue
                total = float(final.get(stay, 50))
                booked = total * (1 - np.exp(-lead / 35)) + rng.normal(0, 1.5)
                rows.append(
                    {
                        "hotel": prop,
                        "stay": stay,
                        "snap": snap,
                        "lead": lead,
                        "booked_rn": max(booked, 0),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def sources_2props():
    dem = _demand_2props()
    cap = _capacity_2props()
    pick = _pickup_2props(dem)
    return dem, cap, pick


def _folds_synthetic(champion: str, n_folds: int = 4) -> pd.DataFrame:
    """fold_metrics sintéticos donde ``champion`` es claramente el mejor."""
    base = {
        "seasonal_naive": (14.0, 0.145, 12.0, 15.5, 0.02),
        "pickup_curve": (11.0, 0.115, 9.5, 12.0, -0.01),
        "ets": (13.0, 0.13, 11.0, 14.0, 0.04),
        "sarimax": (12.5, 0.128, 10.8, 13.5, 0.03),
        "lightgbm": (18.0, 0.17, 15.0, 18.0, 0.18),
        "xgboost": (16.5, 0.156, 14.0, 17.0, 0.15),
        "catboost": (17.0, 0.163, 14.5, 17.5, 0.14),
    }
    base[champion] = (9.0, 0.08, 8.0, 10.0, 0.005)
    rng = np.random.default_rng(42)
    rows = []
    for model, (mape, wmape, mae, rmse, bias) in base.items():
        for fold in range(n_folds):
            rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "mape": mape + rng.normal(0, 0.1),
                    "wmape": wmape + rng.normal(0, 0.001),
                    "mae": mae + rng.normal(0, 0.1),
                    "rmse": rmse + rng.normal(0, 0.1),
                    "bias": bias + rng.normal(0, 0.001),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1.3 — rollout ML corregido
# --------------------------------------------------------------------------

def test_ml_rollout_usa_features_del_dia_a_predecir():
    """Patrón fin de semana exacto: el rollout debe predecir con las features
    as-of del día que se predice, no las del día anterior.

    Con el off-by-one del diagnóstico, la fila de features era la del día
    anterior (``tail(1)``): el sábado se predecía con las features del
    viernes. Aquí el patrón es 100 (fin de semana) / 50 (laboral) y las
    features de calendario son el predictor exacto → un sábado predicho bajo
    o un viernes predicho alto delata el desplazamiento.
    """
    n = 420
    dates = pd.date_range("2025-01-01", periods=n)
    weekend = np.asarray(dates.dayofweek >= 5, dtype=float)
    dem = pd.DataFrame(
        {
            "property": ["corin"] * n,
            "stay_date": dates,
            "room_nights": np.where(weekend, 100.0, 50.0),
        }
    )
    rows = []
    for stay, val in zip(dem["stay_date"], dem["room_nights"]):
        for lead in (30, 7):
            rows.append(
                {
                    "property": "corin",
                    "stay_date": stay,
                    "snapshot_date": stay - pd.Timedelta(days=lead),
                    "lead_days": lead,
                    "pickup_rn": val * 0.5,
                }
            )
    pick = pd.DataFrame(rows)
    c = dates[-1]
    future = pd.date_range(c + pd.Timedelta(days=1), periods=28)

    assert "current_booked_rn" in ML_FEATURES, "los GBM deben usar la señal de pickup"
    fitted = fit_candidate("lightgbm", dem, pick, c, 30)
    assert not fitted.cold_start(), "con 420 estancias en el panel no hay cold start"
    pred = predict_candidate(fitted, pick, future).set_index("stay_date")["raw_forecast_rn"]

    future_weekend = np.asarray(future.dayofweek >= 5)
    assert future_weekend.any() and not future_weekend.all()
    # sábado/lunes del horizonte: features del MISMO día → alto
    assert pred.loc[[d for d, w in zip(future, future_weekend) if w]].min() > 75, (
        "un día de fin de semana se predijo bajo: el rollout usó features del día anterior"
    )
    # viernes/martes: features del MISMO día → bajo
    assert pred.loc[[d for d, w in zip(future, future_weekend) if not w]].max() < 75, (
        "un día laboral se predijo alto: el rollout usó features del día anterior"
    )


# --------------------------------------------------------------------------
# 2.1 — folds trimestrales sin solape + grano por propiedad
# --------------------------------------------------------------------------

def test_folds_no_solapan_ventana_de_reporte():
    from src.data.schema import SourceKind, detect_schema
    from src.forecasting.core import _normalize_demand, _normalize_pickup
    from src.config import load_config

    # histórico largo (desde 2024-06) para que 3+ folds caigan dentro del dato
    dem_raw = _demand_2props(n=500, start="2024-06-01")
    pick_raw = _pickup_2props(dem_raw)
    cfg = load_config()
    dem = _normalize_demand(dem_raw, detect_schema(dem_raw, SourceKind.DEMAND, cfg.column_hints))
    pick = _normalize_pickup(pick_raw, detect_schema(pick_raw, SourceKind.PICKUP, cfg.column_hints))
    cutoff = pd.Timestamp("2025-06-30")
    folds, errors = real_fold_metrics(dem, pick, cutoff)

    assert "property" in folds.columns
    assert (folds["property"] == "PORTFOLIO").any()
    assert set(folds["property"].unique()) >= {"corin", "fiesta", "PORTFOLIO"}
    # separación selección/reporte: ningún error de fold cae en la ventana
    # de reporte (cutoff, cutoff+90]
    assert errors["stay_date"].max() <= cutoff - pd.Timedelta(
        days=FOLD_SPACING - FOLD_HORIZON
    ), "los folds de selección deben terminar antes de la ventana de reporte"
    # cobertura estacional: los 4 folds caen en trimestres distintos
    fold_months = sorted({pd.Timestamp(d).month for d in errors["stay_date"].unique()})
    assert len(fold_months) >= 3, "los folds deben cubrir meses distintos del año"


# --------------------------------------------------------------------------
# 2.3 — Diebold-Mariano: empate → parsimonia
# --------------------------------------------------------------------------

def test_dm_series_identicas_no_significativo():
    actual = np.arange(50, dtype=float)
    fc = actual + 1.0
    dm = diebold_mariano(actual, fc, fc, horizon=30)
    assert not dm.significant
    assert dm.better is None


def test_dm_ventaja_real_significativa():
    rng = np.random.default_rng(2)
    actual = rng.normal(50, 10, 200)
    mejor = actual + rng.normal(0, 1, 200)  # casi perfecto
    peor = actual + rng.normal(0, 12, 200)
    dm = diebold_mariano(actual, mejor, peor, horizon=30)
    assert dm.significant
    assert dm.better == "a"


def test_empate_dm_desempata_por_parsimonia():
    """Sarimax con mejor score pero empate estadístico → gana seasonal_naive."""
    rows = []
    rng = np.random.default_rng(3)
    base = {
        "seasonal_naive": (14.0, 0.115, 12.0, 15.5, 0.02),
        "sarimax": (12.5, 0.108, 10.8, 13.5, 0.03),
    }
    for model, (mape, wmape, mae, rmse, bias) in base.items():
        for fold in range(4):
            rows.append(
                {
                    "model": model, "fold": fold,
                    "mape": mape + rng.normal(0, 0.1),
                    "wmape": wmape + rng.normal(0, 0.001),
                    "mae": mae + rng.normal(0, 0.1),
                    "rmse": rmse + rng.normal(0, 0.1),
                    "bias": bias + rng.normal(0, 0.001),
                }
            )
    folds = pd.DataFrame(rows)
    # error panel: forecasts IDÉNTICOS → DM empate
    dates = pd.date_range("2025-01-01", periods=30)
    errors = []
    for model in ("seasonal_naive", "sarimax"):
        for fold in range(4):
            for d, (a, f) in enumerate(zip(np.full(30, 50.0), np.full(30, 55.0))):
                errors.append(
                    {"model": model, "fold": fold, "stay_date": dates[d], "actual": a, "forecast": f}
                )
    benchmark = run_benchmark(folds, error_panel=pd.DataFrame(errors))
    assert benchmark.champion == "seasonal_naive", (
        "un empate estadístico debe desempatarse por parsimonia hacia el baseline"
    )
    assert "empate estadístico" in benchmark.selection.reason
    md = benchmark_text(benchmark)
    assert "Diebold-Mariano" in md
    assert "empate estadístico" in md


# --------------------------------------------------------------------------
# 3.1/3.2/3.3 — el campeón opera
# --------------------------------------------------------------------------

def test_campeon_inyectado_opera_el_forecast(sources_2props, tmp_path):
    """Folds reales inyectados con ganador seasonal_naive → el forecast lo usa."""
    dem, cap, pick = sources_2props
    folds = _folds_synthetic("seasonal_naive")
    result = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=tmp_path / "champion_ops",
        cutoff=pd.Timestamp("2025-06-30"),
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
        fold_metrics=folds,
    )
    assert result.champion == "seasonal_naive"
    ft = result.forecast_table
    assert (ft["model_name"] == "seasonal_naive").all(), (
        "el forecast debe generarlo el campeón, no un modelo hardcodeado"
    )
    stored = read_manifest(result.manifest_path)
    assert stored["model_version"] == "seasonal_naive-0.1.0"
    assert stored["hyperparameters"]["model"] == "seasonal_naive"
    assert stored["benchmark_source"] == "real_backtest"


def test_seleccion_por_propiedad_opera(sources_2props, tmp_path):
    """Dos propiedades con ganadores distintos → model_name por propiedad."""
    dem, cap, pick = sources_2props
    rows = []
    rng = np.random.default_rng(5)
    for model in ("seasonal_naive", "pickup_curve", "sarimax"):
        for fold in range(4):
            for prop in ("corin", "fiesta"):
                if prop == "corin":
                    wmape = 0.08 if model == "seasonal_naive" else 0.14
                else:
                    wmape = 0.07 if model == "pickup_curve" else 0.13
                rows.append(
                    {
                        "model": model, "fold": fold, "property": prop,
                        "mape": 10.0, "wmape": wmape + rng.normal(0, 0.001),
                        "mae": 10.0, "rmse": 12.0, "bias": 0.01,
                    }
                )
    folds = pd.DataFrame(rows)
    result = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=tmp_path / "per_prop",
        cutoff=pd.Timestamp("2025-06-30"),
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
        fold_metrics=folds,
    )
    ft = result.forecast_table
    by_prop = ft.groupby("property")["model_name"].unique().apply(list)
    assert "seasonal_naive" in by_prop.get("corin", [])
    assert "pickup_curve" in by_prop.get("fiesta", [])
    stored = read_manifest(result.manifest_path)
    assert stored["hyperparameters"]["per_property"]["corin"] == "seasonal_naive"
    assert stored["hyperparameters"]["per_property"]["fiesta"] == "pickup_curve"


def test_campeon_coherente_entre_entregables(sources_2props, tmp_path):
    """Un único campeón en benchmark, manifiesto, tabla y Power BI."""
    dem, cap, pick = sources_2props
    folds = _folds_synthetic("sarimax")
    result = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=tmp_path / "coherence",
        cutoff=pd.Timestamp("2025-06-30"),
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
        fold_metrics=folds,
    )
    champion = result.champion
    assert champion == "sarimax"

    # benchmark md
    bench_md = result.deliverables["model_benchmark"].read_text(encoding="utf-8")
    assert f"**Campeón**: {champion}" in bench_md
    # manifiesto
    stored = read_manifest(result.manifest_path)
    assert stored["model_version"] == f"{champion}-0.1.0"
    assert stored["hyperparameters"]["model"] == champion
    # powerbi/model.csv: un solo campeón y es el del benchmark
    model_csv = pd.read_csv(tmp_path / "coherence" / "powerbi" / "model.csv")
    champions = model_csv.loc[model_csv["is_champion"] == True, "model_name"].tolist()  # noqa: E712
    assert champions == [champion], (
        f"powerbi/model.csv declara {champions}; el benchmark declaró {champion}"
    )
    assert (model_csv["is_champion"] == True).sum() == 1  # noqa: E712
    # la tabla de forecast lo declara por fila
    assert (result.forecast_table["model_name"] == champion).all()


# --------------------------------------------------------------------------
# 4.1 — gate de calidad con dientes
# --------------------------------------------------------------------------

def test_gate_calidad_fail_detiene_la_corrida(sources_2props, tmp_path):
    """RN negativos (error crítico) → la corrida se detiene sin publicar nada."""
    dem, cap, pick = sources_2props
    dem = dem.copy()
    dem.loc[dem.index[5], "rooms_sold"] = -10.0  # error crítico de calidad
    with pytest.raises(CriticalDataError):
        run_end_to_end(
            demand=dem, capacity=cap, pickup=pick,
            run_dir=tmp_path / "gate_fail",
            cutoff=pd.Timestamp("2025-06-30"),
            sample_data=True,
        )
    assert not (tmp_path / "gate_fail" / "future_forecast_dataset.csv").exists()
    assert not (tmp_path / "gate_fail" / "run_manifest.json").exists()


# --------------------------------------------------------------------------
# 4.2 — capacidad futura dentro del perímetro de leakage
# --------------------------------------------------------------------------

def test_capacidad_futura_en_perimetro_es_leakage():
    dates = pd.date_range("2026-01-01", periods=10)
    cap = pd.DataFrame(
        {"property": ["corin"] * 10, "stay_date": dates, "capacity": [100.0] * 10}
    )
    with pytest.raises(LeakageError):
        run_leakage_suite(
            cutoff=pd.Timestamp("2026-01-05"),
            frames={"capacity": (cap, "stay_date")},
        )


# --------------------------------------------------------------------------
# 5.x — bandas ancladas, capadas, calibradas; backtest real
# --------------------------------------------------------------------------

def test_bandas_ancladas_y_capadas_por_lead():
    from src.forecasting.constraints import build_intervals

    dates = pd.date_range("2026-07-01", periods=4)
    df = pd.DataFrame(
        {
            "property": ["corin"] * 4,
            "stay_date": dates,
            "lead_days": [1, 2, 85, 90],
            "raw_forecast_rn": [50.0, 52.0, 55.0, 58.0],
            # el punto publicado ya respetó el techo (final = min(raw, capacity))
            "final_forecast_rn": [50.0, 52.0, 55.0, 56.0],
            "capacity": [60.0, 60.0, 56.0, 56.0],
        }
    )
    residual_bands = {
        "1-2": pd.Series([1.0, -1.0, 1.5, -1.5] * 10),
        "90+": pd.Series([20.0, -25.0, 22.0, -28.0] * 10),
        "60-89": pd.Series([20.0, -25.0, 22.0, -28.0] * 10),
    }
    result = build_intervals(
        df, model_name="x", point_col="final_forecast_rn", residual_bands=residual_bands
    )
    f = result.frame
    # p50 == punto publicado en todos los registros
    assert np.allclose(f["p50_forecast"], f["final_forecast_rn"])
    # p90 jamás excede la capacidad
    assert (f["p90_forecast"] <= f["capacity"] + 1e-9).all()
    assert (f["p10_forecast"] >= 0.0).all()
    # calibración por lead: la banda de leads 90+ es más ancha que la de 1-2
    width_short = (f["p90_forecast"] - f["p10_forecast"]).iloc[:2].mean()
    width_long = (f["p90_forecast"] - f["p10_forecast"]).iloc[2:].mean()
    assert width_long > width_short, "la amplitud debe crecer con el lead"


def test_intervalos_e2e_anclados_y_respetan_capacidad(sources_2props, tmp_path):
    dem, cap, pick = sources_2props
    result = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=tmp_path / "bands",
        cutoff=pd.Timestamp("2025-06-30"),
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
        fold_metrics=_folds_synthetic("pickup_curve"),
    )
    ft = result.forecast_table
    assert np.allclose(ft["p50_forecast"], ft["final_forecast_rn"]), (
        "el p50 debe coincidir con el forecast publicado"
    )
    assert (ft["p90_forecast"] <= ft["capacity"] + 1e-9).all()
    assert (ft["p10_forecast"] <= ft["p50_forecast"] + 1e-9).all()


def test_backtesting_report_con_metricas_reales(sources_2props, tmp_path):
    """El entregable deja de ser un stub: métricas del backtest ejecutado."""
    dem, cap, pick = sources_2props
    result = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=tmp_path / "backtest",
        cutoff=pd.Timestamp("2025-06-30"),
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
        fold_metrics=_folds_synthetic("pickup_curve"),
    )
    report = json.loads(
        result.deliverables["backtesting_report"].read_text(encoding="utf-8")
    )
    assert "see" not in report, "el backtesting report sigue siendo un stub"
    assert report["n_revisions"] > 0
    assert "wmape" in report["global"]
    assert report["by_property"], "debe reportar métricas por propiedad"
    assert report["by_lead_band"], "debe reportar métricas por banda de lead"


def test_backtest_multi_propiedad_no_pisa_actuals():
    """run_backtest por propiedad conserva la dimensión sin pisarse."""
    from src.backtesting.snapshots import run_backtest

    dates = pd.date_range("2026-01-01", periods=40)
    rows = {"property": [], "stay_date": [], "room_nights": []}
    for prop, level in (("corin", 100.0), ("fiesta", 40.0)):
        rows["property"] += [prop] * 40
        rows["stay_date"] += list(dates)
        rows["room_nights"] += [level] * 40
    hist = pd.DataFrame(rows)

    stays = [dates[35]]
    actual_by_prop = {}
    for prop in ("corin", "fiesta"):
        def fn(snapshot, stay, lead, _level=hist.loc[hist["property"] == prop, "room_nights"].iloc[0]):
            return {"forecast_rn": _level, "model_name": "const"}
        bt = run_backtest(
            hist.loc[hist["property"] == prop], stays, fn,
            leads=(7, 1), cache_key=prop,
        )
        actual_by_prop[prop] = bt.actuals[str(stays[0].date())]
    assert actual_by_prop["corin"] == 100.0
    assert actual_by_prop["fiesta"] == 40.0


# --------------------------------------------------------------------------
# 6.x — cobertura, KPIs serializables, drivers reales
# --------------------------------------------------------------------------

def test_accuracy_report_con_cobertura_y_propiedad(sources_2props, tmp_path):
    dem, cap, pick = sources_2props
    # fiesta SIN reales en la ventana evaluada → debe quedar reportada
    dem_eval = dem.copy()
    result = run_end_to_end(
        demand=dem_eval, capacity=cap, pickup=pick,
        run_dir=tmp_path / "coverage",
        cutoff=pd.Timestamp("2025-06-30"),
        actual=dem_eval.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
        fold_metrics=_folds_synthetic("pickup_curve"),
    )
    accuracy = json.loads(
        result.deliverables["forecast_accuracy_report"].read_text(encoding="utf-8")
    )
    assert "global" in accuracy and "wmape" in accuracy["global"]
    assert accuracy["by_property"], "el accuracy report debe persistir el desglose por propiedad"
    coverage = accuracy["evaluation_coverage"]
    assert coverage["n_properties_portfolio"] == 2
    assert coverage["n_properties_evaluated"] >= 1
    assert isinstance(coverage["excluded"], dict)
    # Power BI: desglose por propiedad presente con columna breakdown
    acc_csv = pd.read_csv(tmp_path / "coverage" / "powerbi" / "accuracy.csv")
    assert "breakdown" in acc_csv.columns
    assert (acc_csv["breakdown"] == "property").any()
    assert (acc_csv["breakdown"] == "lead_band").any()


def test_cobertura_reporta_propiedades_excluidas(tmp_path):
    """Una propiedad sin reales en la ventana queda listada como excluida."""
    dem = _demand_2props()
    cap = _capacity_2props()
    pick = _pickup_2props(dem)
    actual = dem.rename(columns={"rooms_sold": "room_nights"})
    actual = actual.loc[actual["property"] == "corin"]  # fiesta sin reales
    result = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=tmp_path / "excluded",
        cutoff=pd.Timestamp("2025-06-30"),
        actual=actual,
        sample_data=True,
        fold_metrics=_folds_synthetic("pickup_curve"),
    )
    accuracy = json.loads(
        result.deliverables["forecast_accuracy_report"].read_text(encoding="utf-8")
    )
    assert accuracy["evaluation_coverage"]["excluded"].get("fiesta")
    stored = read_manifest(result.manifest_path)
    assert stored["evaluation_coverage"]["n_properties_evaluated"] == 1


def test_manifest_sin_nan_literal(sources_2props, tmp_path):
    """Los KPIs sin dato se serializan como null, nunca como NaN."""
    dem, cap, pick = sources_2props
    result = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=tmp_path / "no_nan",
        cutoff=pd.Timestamp("2025-06-30"),
        actual=None,  # sin reales → forecast_vs_previous queda sin dato
        sample_data=True,
        fold_metrics=_folds_synthetic("pickup_curve"),
    )
    raw = result.manifest_path.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw

    def _reject(value: str):
        raise ValueError(f"constante JSON inválida: {value}")

    payload = json.loads(raw, parse_constant=_reject)  # NaN/Infinity romperían aquí
    assert payload["kpis"]["forecast_vs_previous"] is None


def test_explainability_drivers_reales_no_fijos():
    """Los drivers derivan de los valores de la fila, no de pesos fijos."""
    fc = pd.DataFrame(
        {
            "property": ["corin", "corin"],
            "stay_date": pd.date_range("2026-07-01", periods=2),
            "current_booked_rn": [40.0, 5.0],
            "expected_pickup_rn": [10.0, 45.0],
            "capacity": [50.0, 100.0],
            "raw_forecast_rn": [50.0, 50.0],
            "final_forecast_rn": [50.0, 50.0],
        }
    )
    out = _explain_forecast(fc, champion="pickup_curve")
    assert len(out) == 2
    # la fila con cap (raw == capacity) y la fila sin cap difieren en drivers
    assert out["top_drivers"].iloc[0] != out["top_drivers"].iloc[1], (
        "los drivers deben cambiar cuando cambian los valores de la fila"
    )
    assert (out["champion"] == "pickup_curve").all()


# --------------------------------------------------------------------------
# monitoring en la corrida productiva
# --------------------------------------------------------------------------

def test_monitoring_registrado_en_manifiesto(sources_2props, tmp_path):
    dem, cap, pick = sources_2props
    result = run_end_to_end(
        demand=dem, capacity=cap, pickup=pick,
        run_dir=tmp_path / "monitoring",
        cutoff=pd.Timestamp("2025-06-30"),
        actual=dem.rename(columns={"rooms_sold": "room_nights"}),
        sample_data=True,
        fold_metrics=_folds_synthetic("pickup_curve"),
    )
    stored = read_manifest(result.manifest_path)
    monitoring = stored.get("monitoring", {})
    assert "n_outliers" in monitoring
    assert "retraining_decision" in monitoring
    assert "should_retrain" in monitoring["retraining_decision"]
