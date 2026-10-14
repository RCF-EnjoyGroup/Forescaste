"""Tests de reporting (tasks 9.1–9.7)."""

import numpy as np
import pandas as pd
import pytest

from src.reporting.benchmark import (
    CANDIDATE_ORDER,
    DELIVERABLES,
    DeliverablesError,
    benchmark_text,
    generate_deliverables,
    run_benchmark,
)
from src.reporting.forecast_table import (
    EXTENDED_COLUMNS,
    REQUIRED_COLUMNS,
    build_forecast_table,
    build_powerbi_dataset,
    executive_kpis,
    extended_presence,
    validate_forecast_table,
)


def _forecast_df(with_intervals=True, with_extended=True):
    n = 10
    df = pd.DataFrame(
        {
            "property": ["corin"] * n,
            "stay_date": pd.date_range("2026-03-01", periods=n),
            "forecast_date": pd.Timestamp("2026-01-01"),
            "lead_days": [59] * n,
            "current_booked_rn": np.linspace(30, 50, n),
            "capacity": [100.0] * n,
            "raw_forecast_rn": np.linspace(55, 95, n),
            "final_forecast_rn": np.linspace(55, 95, n),
            "model_name": ["lightgbm"] * n,
            "model_version": ["0.1.0"] * n,
            "forecast_run_timestamp": ["2026-01-01T00:00:00+00:00"] * n,
        }
    )
    if with_intervals:
        df["p10_forecast"] = df["raw_forecast_rn"] - 8
        df["p50_forecast"] = df["raw_forecast_rn"]
        df["p90_forecast"] = df["raw_forecast_rn"] + 8
    if with_extended:
        df["historical_same_day_rn"] = np.linspace(50, 90, n)
        df["expected_pickup_rn"] = 5.0
        df["pickup_rate"] = 0.1
        df["forecast_occupancy"] = df["final_forecast_rn"] / 100
        df["forecast_bias"] = 0.02
        df["confidence_level"] = 0.8
    return df


# --------------------------- 9.1 ----------------------------------------


def test_tabla_final_columnas_obligatorias():
    table = build_forecast_table(_forecast_df())
    for col in REQUIRED_COLUMNS:
        assert col in table.columns, f"falta columna obligatoria {col}"
    problems = validate_forecast_table(table)
    assert problems == [], f"tabla con problemas: {problems}"
    assert len(table) == 10
    # pobladas
    assert table["property"].notna().all()
    assert table["stay_date"].notna().all()
    assert table["raw_forecast_rn"].notna().all()
    assert table["model_name"].notna().all()


def test_tabla_final_sin_columnas_obligatorias_las_crea_y_reporta():
    partial = pd.DataFrame(
        {
            "property": ["corin"],
            "stay_date": [pd.Timestamp("2026-03-01")],
            "raw_forecast_rn": [50.0],
            "capacity": [100.0],
            "final_forecast_rn": [50.0],
        }
    )
    table = build_forecast_table(partial)
    for col in REQUIRED_COLUMNS:
        assert col in table.columns
    assert table["_missing_required"].iloc[0] != ""
    assert "p10_forecast" in table["_missing_required"].iloc[0]

    problems = validate_forecast_table(table)
    assert any("nulos" in p for p in problems), "los obligatorios ausentes deben reportarse"


# --------------------------- 9.2 ----------------------------------------


def test_columnas_extendidas_presentes():
    table = build_forecast_table(_forecast_df(with_extended=True))
    presence = extended_presence(table)
    assert all(presence.values()), "todas las extendidas provistas deben estar"
    for col in EXTENDED_COLUMNS:
        assert col in table.columns
    assert table["_present_extended"].iloc[0] != ""


def test_columnas_extendidas_ausentes_no_inventadas():
    table = build_forecast_table(_forecast_df(with_extended=False))
    presence = extended_presence(table)
    # el build solo agrega las presentes en la entrada
    for col in EXTENDED_COLUMNS:
        assert presence[col] == (col in table.columns)
    assert not any(presence.values())
    # pero la tabla sigue siendo válida (extendidas son opcionales)
    problems = validate_forecast_table(table)
    assert problems == []


# --------------------------- 9.3 Power BI -------------------------------


def test_powerbi_cuatro_areas_cargables():
    fc = _forecast_df()
    fc["actual_rn"] = fc["final_forecast_rn"] - 5

    accuracy = pd.DataFrame(
        {
            "property": ["corin"],
            "stay_date": ["2026-03-01"],
            "lead_band": ["30-59"],
            "mape": [8.5],
            "wmape": [0.08],
            "mae": [4.0],
            "rmse": [5.5],
            "bias": [0.02],
        }
    )
    pickup = pd.DataFrame(
        {
            "property": ["corin"],
            "stay_date": [pd.Timestamp("2026-03-01")],
            "current_booked_rn": [50.0],
            "expected_pickup_rn": [10.0],
            "lead_days": [30],
        }
    )
    models = pd.DataFrame(
        {"model_name": ["lightgbm", "prophet"], "model_version": ["0.1", "0.2"],
         "is_champion": [True, False], "is_challenger": [False, True]}
    )

    ds = build_powerbi_dataset(fc, accuracy, pickup, models)
    assert set(ds) == {"forecast", "accuracy", "pickup", "model"}

    # área forecast con Variance y Variance %
    f = ds["forecast"]
    assert "variance" in f.columns and "variance_pct" in f.columns
    assert f["variance"].notna().all()
    assert (f["variance"] == 5).all()  # final - actual = 5

    # área accuracy con las 5 métricas
    a = ds["accuracy"]
    assert {"mape", "wmape", "mae", "rmse", "bias"} <= set(a.columns)
    assert a["mape"].iloc[0] == 8.5

    # área pickup con pickup_pct calculado
    p = ds["pickup"]
    assert "pickup_pct" in p.columns
    assert p["pickup_pct"].iloc[0] == pytest.approx(10 / 60)

    # área modelo con champion/challenger
    m = ds["model"]
    assert {"model_name", "model_version", "is_champion", "is_challenger"} <= set(m.columns)


def test_powerbi_sin_actuales_variance_nan():
    ds = build_powerbi_dataset(_forecast_df())
    assert "variance" in ds["forecast"].columns
    assert ds["forecast"]["variance"].isna().all()


# --------------------------- 9.4 KPIs -----------------------------------


def test_siete_kpis_calculados():
    fc = _forecast_df()
    actual = pd.DataFrame(
        {
            "property": ["corin"] * 10,
            "stay_date": pd.date_range("2026-03-01", periods=10),
            "room_nights": fc["final_forecast_rn"] - 3,
        }
    )
    prev = _forecast_df()
    prev["final_forecast_rn"] = prev["final_forecast_rn"] * 0.95

    kpis = executive_kpis(
        fc,
        actual=actual,
        previous_forecast=prev,
        expected_pickup_total=50.0,
    )

    # los 7 KPIs requeridos
    required = [
        "forecast_room_nights",
        "forecast_occupancy",
        "forecast_accuracy_wmape",
        "forecast_accuracy_mape",
        "forecast_bias",
        "expected_pickup",
        "forecast_vs_previous",
        "capacity_utilization",
    ]
    for k in required:
        assert k in kpis, f"falta KPI {k}"

    assert np.isfinite(kpis["forecast_room_nights"])
    assert kpis["forecast_room_nights"] == pytest.approx(fc["final_forecast_rn"].sum())
    assert 0 <= kpis["forecast_occupancy"] <= 1
    assert np.isfinite(kpis["forecast_accuracy_wmape"])
    assert np.isfinite(kpis["forecast_bias"])
    assert kpis["expected_pickup"] == 50.0
    assert np.isfinite(kpis["forecast_vs_previous"])
    assert kpis["forecast_vs_previous"] > 0  # actual > previo


def test_kpis_sin_actuales_nan_documentado():
    kpis = executive_kpis(_forecast_df())
    assert np.isnan(kpis["forecast_accuracy_wmape"])
    assert np.isnan(kpis["forecast_bias"])
    assert kpis["mape_excluded_zero_actual"] == 0.0
    assert np.isfinite(kpis["forecast_room_nights"])


# --------------------------- 9.5 benchmark ------------------------------


def _fold_metrics():
    rows = []
    rng = np.random.default_rng(3)
    base = {
        "seasonal_naive": (14.0, 0.145, 12.0, 0.02),
        "pickup_curve": (11.0, 0.115, 9.5, -0.01),
        "ets": (13.0, 0.13, 11.0, 0.04),
        "sarimax": (12.5, 0.128, 10.8, 0.03),
        "lightgbm": (9.0, 0.12, 10.0, 0.18),
        "xgboost": (9.5, 0.125, 10.2, 0.15),
        "catboost": (9.8, 0.126, 10.4, 0.14),
    }
    for model, (mape, wmape, mae, bias) in base.items():
        for fold in range(4):
            rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "mape": mape + rng.normal(0, 0.3),
                    "wmape": wmape + rng.normal(0, 0.004),
                    "mae": mae + rng.normal(0, 0.3),
                    "rmse": mae * 1.25 + rng.normal(0, 0.3),  # RMSE >= MAE
                    "bias": bias + rng.normal(0, 0.01),
                }
            )
    return pd.DataFrame(rows)


def test_benchmark_tabla_7_candidatos():
    result = run_benchmark(_fold_metrics())
    table = result.table

    assert len(table) == 7
    assert list(table["model"]) == CANDIDATE_ORDER
    # 6 métricas del spec + extras
    for col in ("mape", "wmape", "mae", "rmse", "bias", "stability"):
        assert col in table.columns, f"falta columna {col}"
        assert table[col].notna().all()

    # campeón declarado con evidencia
    assert table["is_champion"].sum() == 1
    assert table.loc[table["is_champion"], "model"].iloc[0] == result.champion
    assert "no decide" in result.selection.reason
    # el campeón es el de mejor combinación (pickup_curve), no el de mejor MAPE
    assert result.champion == "pickup_curve"

    # accuracy band interpretativo
    assert "accuracy_band" in table.columns


def test_benchmark_texto_legible():
    result = run_benchmark(_fold_metrics())
    text = benchmark_text(result)
    assert text.startswith("| model |")
    assert text.count("\n") >= 9  # header + sep + 7 filas
    assert "**Campeón**" in text
    for m in CANDIDATE_ORDER:
        assert m in text


# --------------------------- 9.6 entregables ----------------------------


def test_entregables_completos(tmp_path):
    from src.explainability.explainer import explain_from_importance

    expl = explain_from_importance(
        property="corin", stay_date="2026-03-01", forecast_rn=82.0,
        feature_values={"current_booked_rn": 54.0},
        feature_importance={"current_booked_rn": 1.0},
    )
    explain_df = pd.DataFrame(
        [{"property": expl.property, "stay_date": expl.stay_date,
          "drivers": str(expl.top_drivers())}]
    )
    benchmark = run_benchmark(_fold_metrics())
    powerbi = build_powerbi_dataset(_forecast_df())

    paths = generate_deliverables(
        tmp_path / "run1",
        data_quality_report={"status": "pass"},
        eda_report={"n_rows": 1000},
        feature_report={"n_features": 42},
        benchmark=benchmark,
        backtest_report={"n_snapshots": 21},
        accuracy_report={"wmape": 0.11},
        forecast_table=build_forecast_table(_forecast_df()),
        explainability=explain_df,
        powerbi=powerbi,
        technical_doc="# Doc\nLimitaciones: ...",
    )

    assert set(paths) == set(DELIVERABLES)
    for name, path in paths.items():
        assert path.exists(), f"entregable {name} no escrito"
    # los 10 con contenido
    assert (tmp_path / "run1" / "model_benchmark.md").stat().st_size > 0
    assert (tmp_path / "run1" / "future_forecast_dataset.csv").stat().st_size > 0
    assert (tmp_path / "run1" / "powerbi" / "forecast.csv").exists()


def test_entregables_incompletos_fallan(tmp_path):
    """Si falta un entregable, la corrida falla (no se publica a medias)."""
    with pytest.raises(DeliverablesError) as exc:
        generate_deliverables(tmp_path / "run2")  # nada provisto
    msg = str(exc.value)
    assert "faltan entregables" in msg
    for name in DELIVERABLES[:3]:
        assert name in msg
    # no escribió nada
    assert not (tmp_path / "run2" / "model_benchmark.md").exists()


def test_entregables_parcial_falla(tmp_path):
    from src.explainability.explainer import explain_from_importance

    expl = explain_from_importance(
        property="corin", stay_date="2026-03-01", forecast_rn=82.0,
        feature_values={"a": 1.0}, feature_importance={"a": 1.0},
    )
    with pytest.raises(DeliverablesError, match="faltan entregables"):
        generate_deliverables(
            tmp_path / "run3",
            data_quality_report={"status": "pass"},
            eda_report={},
            feature_report={},
            benchmark=run_benchmark(_fold_metrics()),
            backtest_report={},
            accuracy_report={},
            forecast_table=build_forecast_table(_forecast_df()),
            explainability=pd.DataFrame([{"property": "corin"}]),
            powerbi=build_powerbi_dataset(_forecast_df()),
            # technical_doc falta
        )


# --------------------------- 9.7 ingeniería -----------------------------


def test_estandar_ingenieria_logica_en_src():
    """9.7: la lógica vive en src/, los tests solo consumen la interfaz."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    src = root / "src"
    assert src.exists()

    # todos los módulos del paquete existen y son importables
    expected = [
        "data", "validation", "features", "forecasting", "models",
        "evaluation", "backtesting", "explainability", "monitoring", "reporting",
    ]
    for mod in expected:
        assert (src / mod / "__init__.py").exists(), f"falta subpaquete src/{mod}"

    # los módulos de lógica no son notebooks
    py_files = list(src.rglob("*.py"))
    assert len(py_files) >= 15
    assert not list(src.rglob("*.ipynb")), "la lógica no debe vivir en notebooks"


# ------------------------ reporte visual HTML ---------------------------

from src.reporting.visual_report import generate_visual_report  # noqa: E402


def test_reporte_visual_genera_html_autocontenido(tmp_path):
    fc = _forecast_df()
    out = generate_visual_report(tmp_path, fc, benchmark=run_benchmark(_fold_metrics()).table)
    assert out.name == "report.html" and out.exists()
    html = out.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html, "los gráficos van embebidos en base64"
    assert "Forecast" in html


def test_reporte_visual_no_requiere_datos_opcionales(tmp_path):
    # sin benchmark, sin demand, sin métricas: no falla, solo omite secciones
    out = generate_visual_report(tmp_path, _forecast_df())
    assert out.exists()


def test_reporte_visual_deriva_lead_band(tmp_path):
    fc = _forecast_df()  # trae lead_days=59, no lead_band
    out = generate_visual_report(tmp_path, fc)
    assert out.exists()
    assert "Pickup" in out.read_text(encoding="utf-8")


def test_reporte_visual_ignora_actual_de_longitud_distinta(tmp_path):
    fc = _forecast_df()
    bad = pd.Series([1.0, 2.0])  # largo != len(fc)
    out = generate_visual_report(tmp_path, fc, actual=bad)
    assert out.exists(), "un actual mal alineado no debe romper el reporte"

# --------------------------------------------------------------------------
# cambio enhance-visual-report: secciones por propiedad
# --------------------------------------------------------------------------


def _fc_multi_propiedad(n=20):
    dates = pd.date_range("2026-03-01", periods=n)
    frames = []
    for prop, base in (("CORIN", 90.0), ("FIESTA", 40.0)):
        frames.append(
            pd.DataFrame(
                {
                    "property": [prop] * n,
                    "stay_date": dates,
                    "forecast_date": pd.Timestamp("2026-02-01"),
                    "lead_days": list(range(29, 9, -1))[:n] if n <= 20 else [59] * n,
                    "current_booked_rn": np.linspace(base * 0.3, base * 0.7, n),
                    "capacity": [base * 1.2] * n,
                    "raw_forecast_rn": np.linspace(base * 0.6, base, n),
                    "final_forecast_rn": np.linspace(base * 0.6, base, n),
                    "model_name": ["pickup_curve"] * n,
                    "model_version": ["0.1.0"] * n,
                    "forecast_run_timestamp": ["2026-02-01T00:00:00+00:00"] * n,
                    "p10_forecast": np.linspace(base * 0.5, base * 0.9, n),
                    "p50_forecast": np.linspace(base * 0.6, base, n),
                    "p90_forecast": np.linspace(base * 0.7, base * 1.1, n),
                    "forecast_occupancy": np.linspace(0.5, 0.8, n),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _property_breakdown_df():
    return pd.DataFrame(
        {
            "property": ["CORIN", "FIESTA"],
            "mape": [12.5, 22.0],
            "wmape": [0.12, 0.22],
            "mae": [8.0, 6.0],
            "rmse": [11.0, 8.5],
            "bias": [-0.03, 0.05],
            "n_obs": [20, 20],
            "rank": [1, 2],
        }
    )


def _backtest_report_dict():
    return {
        "n_revisions": 28,
        "global": {"mape": 20.0, "wmape": 0.2},
        "by_property": [
            {"property": "CORIN", "model": "pickup_curve", "n_obs": 14,
             "mape": 18.0, "wmape": 0.18},
            {"property": "FIESTA", "model": "sarimax", "n_obs": 14,
             "mape": 25.0, "wmape": 0.25},
        ],
        "by_lead_band": [
            {"lead_band": "90+", "n_obs": 4, "mape": 40.0, "wmape": 0.40},
            {"lead_band": "30-59", "n_obs": 4, "mape": 25.0, "wmape": 0.25},
            {"lead_band": "1-2", "n_obs": 4, "mape": 10.0, "wmape": 0.10},
        ],
        "by_property_lead_band": [
            {"property": "CORIN", "lead_band": "90+", "n_obs": 2, "mape": 35.0, "wmape": 0.35},
            {"property": "CORIN", "lead_band": "30-59", "n_obs": 2, "mape": 20.0, "wmape": 0.20},
            {"property": "CORIN", "lead_band": "1-2", "n_obs": 2, "mape": 8.0, "wmape": 0.08},
            {"property": "FIESTA", "lead_band": "90+", "n_obs": 2, "mape": 45.0, "wmape": 0.45},
            {"property": "FIESTA", "lead_band": "30-59", "n_obs": 2, "mape": 30.0, "wmape": 0.30},
            {"property": "FIESTA", "lead_band": "1-2", "n_obs": 2, "mape": 12.0, "wmape": 0.12},
        ],
    }


def test_reporte_visual_con_detalle_por_propiedad(tmp_path):
    fc = _fc_multi_propiedad()
    kpis = {
        "forecast_room_nights": 2600.0,
        "forecast_occupancy": 0.6,
        "n_properties_evaluated": 2.0,
        "n_properties_portfolio": 2.0,
    }
    out = generate_visual_report(
        tmp_path, fc,
        kpis=kpis,
        property_breakdown=_property_breakdown_df(),
        model_by_property={"CORIN": "pickup_curve", "FIESTA": "sarimax"},
        backtest_report=_backtest_report_dict(),
        champion="pickup_curve",
    )
    html = out.read_text(encoding="utf-8")
    # secciones nuevas presentes
    assert "Forecast por propiedad" in html
    assert "Accuracy por propiedad" in html
    assert "Revisiones T-90..T-1" in html
    # KPIs nuevos: campeón y propiedades evaluadas n/N
    assert "Campe" in html and "pickup_curve" in html
    assert "2 / 2" in html
    # la tabla de accuracy declara el modelo de cada propiedad
    assert "sarimax" in html
    # imágenes embebidas de las secciones nuevas
    assert html.count("data:image/png;base64,") >= 4


def test_reporte_visual_por_propiedad_se_omita_sin_datos(tmp_path):
    fc = _forecast_df()  # una sola propiedad, sin desgloses ni backtest
    out = generate_visual_report(tmp_path, fc)
    html = out.read_text(encoding="utf-8")
    assert "Accuracy por propiedad" not in html
    assert "Revisiones T-90..T-1" not in html
    assert "Campe" not in html  # sin champion no hay tarjeta


def test_reporte_visual_accuracy_con_grano_semanal_y_advertencia(tmp_path):
    """La tabla muestra WMAPE semanal y marca el MAPE con P10 < 10 RN."""
    import pandas as pd

    fc = _fc_multi_propiedad()
    daily = pd.DataFrame(
        {
            "property": ["SJOSL", "FIESTA"],
            "wmape": [0.34, 0.12], "mape": [51.0, 12.0],
            "mae": [7.0, 40.0], "rmse": [9.0, 50.0], "bias": [0.0, -0.1],
            "n_obs": [90, 90], "n_mape_excluded": [0, 0],
            "mean_actual": [20.6, 362.0], "p10_actual": [7.0, 300.0],
            "rank": [2, 1],
        }
    )
    weekly = pd.DataFrame(
        {
            "property": ["SJOSL", "FIESTA"],
            "wmape": [0.215, 0.06], "mape": [24.7, 6.0],
            "mae": [30.0, 2500.0], "rmse": [35.0, 3000.0], "bias": [0.0, -0.05],
            "n_obs": [13, 13], "n_mape_excluded": [0, 0],
        }
    )
    out = generate_visual_report(
        tmp_path, fc,
        property_breakdown=daily,
        property_weekly_breakdown=weekly,
        model_by_property={"SJOSL": "pickup_curve", "FIESTA": "lightgbm"},
    )
    html = out.read_text(encoding="utf-8")
    assert "WMAPE sem" in html, "la columna semanal debe aparecer"
    assert "0.215" in html, "el WMAPE semanal de SJOSL debe verse"
    assert "51.0†" in html, "el MAPE de la propiedad con P10<10 lleva la marca"
    assert "12.0†" not in html, "la propiedad grande no lleva marca"
    assert "P10 < 10 RN" in html or "P10 < 10 RN" in html, "la nota explica la marca"
    # sin weekly: la columna existe pero la celda queda con —; la marca de MAPE no depende del weekly
    html_solo = generate_visual_report(
        tmp_path, fc, property_breakdown=daily,
        model_by_property={"SJOSL": "pickup_curve"},
    ).read_text(encoding="utf-8")
    assert "WMAPE sem" in html_solo, "la columna existe aunque falte el desglose"
    assert "51.0†" in html_solo, "la advertencia de MAPE no depende del weekly"


def test_sampled_backtest_emite_desglose_por_propiedad_y_banda():
    from src.forecasting.runner import _sampled_backtest
    from tests.test_remediation import _demand_2props, _pickup_2props

    dem = _demand_2props(n=400)
    pick = _pickup_2props(dem)
    history = dem.rename(columns={"rooms_sold": "room_nights"})
    pickup = pick.rename(
        columns={
            "hotel": "property", "stay": "stay_date", "snap": "snapshot_date",
            "lead": "lead_days", "booked_rn": "pickup_rn",
        }
    )
    report, bands, residuals = _sampled_backtest(
        history, pickup,
        {"corin": "seasonal_naive", "fiesta": "seasonal_naive"},
        pd.Timestamp("2026-06-20"),
        backtest_cfg={"n_stays": 4, "lookback_days": 30},
    )
    rows = report.get("by_property_lead_band") or []
    assert rows, "el backtest debe emitir el desglose propiedad × banda"
    keys = {(r["property"], r["lead_band"]) for r in rows}
    assert ("corin", "1-2") in keys and ("fiesta", "90+") in keys
    for r in rows:
        assert {"property", "lead_band", "n_obs", "wmape", "mape"} <= set(r)
