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
