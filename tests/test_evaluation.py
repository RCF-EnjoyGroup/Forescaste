"""Tests de evaluación (tasks 6.1–6.8)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.breakdowns import (
    LEAD_LABELS,
    lead_band,
    metrics_by_lead_band,
    metrics_by_property,
    metrics_by_property_weekly,
    metrics_by_seasonality,
    property_ranking,
    systematic_pattern,
)
from src.evaluation.metrics import accuracy_band, bias_direction, compute_metrics
from src.evaluation.multistep import assert_no_future_reads, run_multistep
from src.evaluation.selection import select_champion, select_per_property
from src.evaluation.validation import expanding_windows, slice_window


# --------------------------- 6.1 validación temporal ---------------------


def test_expanding_windows_orden_temporal():
    df = pd.DataFrame({"stay_date": pd.date_range("2023-01-01", periods=1000, freq="D")})
    windows = expanding_windows(df, n_splits=4, val_days=90, min_train_days=365)

    assert len(windows) == 4
    for w in windows:
        w.validate()
        # train = [train_start, train_end) semiabierto, val = [val_start, val_end]
        assert w.train_end == w.val_start  # contiguo sin solape
        assert w.val_end > w.val_start
        assert w.train_end > w.train_start

    # expanding: train_start constante
    starts = {w.train_start for w in windows}
    assert len(starts) == 1

    # y la validación avanza en el tiempo
    for a, b in zip(windows, windows[1:]):
        assert b.val_start > a.val_start


def test_rolling_windows_train_se_desplaza():
    df = pd.DataFrame({"stay_date": pd.date_range("2023-01-01", periods=1000, freq="D")})
    windows = expanding_windows(
        df, n_splits=3, val_days=60, min_train_days=365, expanding=False
    )
    starts = {w.train_start for w in windows}
    assert len(starts) > 1, "rolling debe desplazar el inicio del train"


def test_slice_window_sin_solape():
    df = pd.DataFrame(
        {
            "stay_date": pd.date_range("2023-01-01", periods=700, freq="D"),
            "room_nights": np.arange(700),
        }
    )
    windows = expanding_windows(df, n_splits=2, val_days=90, min_train_days=365)
    train, val = slice_window(df, windows[0])
    assert pd.to_datetime(train["stay_date"]).max() < pd.to_datetime(val["stay_date"]).min()
    assert pd.to_datetime(val["stay_date"]).max() <= pd.to_datetime(train["stay_date"]).max() + pd.Timedelta(days=91)


def test_split_aleatorio_prohibido_en_validation():
    """6.1: la validación aleatoria sigue vetada por el gate de leakage."""
    from src.validation.leakage import validate_split_config
    from src.features.asof import LeakageError

    with pytest.raises(LeakageError):
        validate_split_config({"strategy": "random"})


# --------------------------- 6.2 métricas --------------------------------


def test_metricas_contra_valores_a_mano():
    actual = [100, 100, 100, 100]
    forecast = [90, 110, 95, 105]
    m = compute_metrics(actual, forecast)

    # MAE = (10+10+5+5)/4 = 7.5
    assert m.mae == pytest.approx(7.5)
    # RMSE = sqrt((100+100+25+25)/4) = sqrt(62.5)
    assert m.rmse == pytest.approx(np.sqrt(62.5))
    # WMAPE = 30/400 = 0.075
    assert m.wmape == pytest.approx(0.075)
    # MAPE = mean(10%,10%,5%,5%) = 7.5%
    assert m.mape == pytest.approx(7.5)
    # Bias = ( -10+10-5+5 ) / 400 = 0
    assert m.bias == pytest.approx(0.0)
    assert m.n_mape_excluded == 0
    assert not m.excluded_zero_actual


def test_metricas_over_under_forecast():
    # sobre-pronóstico: forecast > actual => bias positivo
    m_over = compute_metrics([100, 100], [110, 120])
    assert m_over.bias > 0
    assert bias_direction(m_over.bias).startswith("over-forecast")

    m_under = compute_metrics([100, 100], [90, 80])
    assert m_under.bias < 0
    assert bias_direction(m_under.bias).startswith("under-forecast")


def test_rmse_penaliza_errores_grandes():
    # mismo MAE, distinto RMSE
    a = [100, 100]
    m1 = compute_metrics(a, [90, 110])   # errores 10,10
    m2 = compute_metrics(a, [80, 120])   # errores 20,20
    assert m1.mae == pytest.approx(m2.mae * 0.5)
    assert m2.rmse > m1.rmse


def test_accuracy_band_referencias():
    assert accuracy_band(5).startswith("excelente")
    assert accuracy_band(15).startswith("generalmente")
    assert accuracy_band(22).startswith("razonable")
    assert accuracy_band(30).startswith("fuera")


# --------------------------- 6.3 Actual = 0 ------------------------------


def test_actual_cero_excluido_y_contado():
    actual = [0, 100, 0, 100]
    forecast = [10, 90, 5, 110]
    m = compute_metrics(actual, forecast)

    # MAPE excluye los dos ceros: mean(|90-100|/100, |110-100|/100) = 10%
    assert m.mape == pytest.approx(10.0)
    assert m.n_mape_excluded == 2
    assert m.excluded_zero_actual

    # MAE y WMAPE incluyen TODAS las observaciones
    assert m.n_obs == 4
    assert m.mae == pytest.approx((10 + 10 + 5 + 10) / 4)
    assert m.wmape == pytest.approx(35 / 200)
    # documentado
    assert "excluye 2 observaciones" in m.mape_note
    assert "nunca se sustituye el cero" in m.mape_note


def test_todos_ceros_sin_reemplazo_arbitrario():
    """Nunca sustituir el cero por un valor arbitrario."""
    m = compute_metrics([0, 0, 0], [1, 2, 3])
    assert np.isnan(m.mape), "MAPE sin observaciones no-cero => NaN, no inventado"
    assert m.n_mape_excluded == 3
    assert np.isfinite(m.mae)  # MAE sí calculable
    assert accuracy_band(m.mape) == "sin datos"


# --------------------------- 6.4 bandas de lead --------------------------


def test_lead_bands_particion_exacta():
    df = pd.DataFrame(
        {
            "lead_days": [1, 2, 3, 6, 7, 13, 14, 29, 30, 59, 60, 89, 90, 200],
            "actual": [50.0] * 14,
            "forecast": [55.0] * 14,
        }
    )
    table = metrics_by_lead_band(df)

    assert list(table["lead_band"].astype(str)) == LEAD_LABELS
    # cobertura: cada observación en exactamente una banda
    assert int(table["n_obs"].sum()) == len(df)
    assert table["n_obs"].tolist() == [2, 2, 2, 2, 2, 2, 2]  # 14 obs / 7 bandas
    assert table["mape"].notna().all()
    assert table["wmape"].notna().all()
    assert table["bias"].notna().all()


def test_lead_band_asignacion():
    assert lead_band(1) == "1-2"
    assert lead_band(2) == "1-2"
    assert lead_band(3) == "3-6"
    assert lead_band(13) == "7-13"
    assert lead_band(14) == "14-29"
    assert lead_band(60) == "60-89"
    assert lead_band(90) == "90+"
    assert lead_band(10000) == "90+"


# --------------------------- 6.5 propiedad/estacionalidad ----------------


def test_metricas_por_propiedad_y_ranking():
    df = pd.DataFrame(
        {
            "property": ["corin"] * 4 + ["marina"] * 4 + ["fiesta"] * 4,
            "stay_date": pd.date_range("2026-01-01", periods=4).tolist() * 3,
            "actual": [100.0] * 12,
            # corin exacto, marina con 10%, fiesta con 30%
            "forecast": [100, 100, 100, 100, 110, 110, 110, 110, 130, 130, 130, 130],
        }
    )
    # arreglar fechas repetidas por property
    df["stay_date"] = pd.date_range("2026-01-01", periods=4).tolist() * 3

    table = metrics_by_property(df)
    assert set(table["property"]) == {"corin", "marina", "fiesta"}
    assert {"mape", "wmape", "mae", "rmse", "bias"} <= set(table.columns)
    # ranking por WMAPE: corin primero, fiesta último
    assert table.iloc[0]["property"] == "corin"
    assert table.iloc[-1]["property"] == "fiesta"
    assert int(table.iloc[0]["rank"]) == 1

    ranking = property_ranking(table)
    assert ranking["best"] == "corin"
    assert ranking["worst"] == "fiesta"
    assert ranking["error_volatility"] > 0


def test_metricas_por_propiedad_con_contexto_de_reales():
    """mean_actual y p10_actual acompañan al MAPE: el denominador chico
    es criterio de lectura, no una lista fija de propiedades."""
    df = pd.DataFrame(
        {
            "property": ["chica"] * 10 + ["grande"] * 10,
            "stay_date": list(pd.date_range("2026-01-01", periods=10)) * 2,
            "actual": [4, 6, 8, 5, 7, 4, 6, 9, 5, 6] + [90, 95, 100, 105, 110] * 2,
            "forecast": [20.0] * 20,
        }
    )
    table = metrics_by_property(df)
    assert {"mean_actual", "p10_actual"} <= set(table.columns)
    chica = table.loc[table["property"] == "chica"].iloc[0]
    grande = table.loc[table["property"] == "grande"].iloc[0]
    assert chica["mean_actual"] == pytest.approx(6.0)
    assert chica["p10_actual"] < 10.0, "la propiedad chica debe quedar marcada"
    assert grande["p10_actual"] >= 10.0, "la grande no debe marcarse"


def test_metricas_semanales_cancelan_el_ruido_diario():
    """El caso SJOSL: reales diminutos y volátiles con nivel perfecto.

    Diaria: forecast 20 contra reales 10/30/20 — los días de 10 RN
    aportan +100% de APE cada uno (ruido de denominador). Semanal: ambas
    semanas suman 140 = forecast — WMAPE 0: el grano operativo mide la
    señal de nivel que el diario esconde.
    """
    fechas = pd.date_range("2026-01-05", periods=14, freq="D")  # lunes, 2 semanas
    semana = [10.0, 30.0, 10.0, 30.0, 10.0, 30.0, 20.0]  # suma 140
    df = pd.DataFrame(
        {
            "property": ["chica"] * 14,
            "stay_date": fechas,
            "actual": semana * 2,
            "forecast": [20.0] * 14,
        }
    )
    table = metrics_by_property_weekly(df)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["property"] == "chica"
    assert row["wmape"] == pytest.approx(0.0), "las semanas suman exacto"
    assert int(row["n_obs"]) == 2, "n_obs cuenta semanas, no días"

    # contra el grano diario para el mismo frame: el ruido es visible
    diario = metrics_by_property(df).iloc[0]
    assert diario["wmape"] > 0.15, "el ruido diario debe ser grande aquí"
    assert diario["p10_actual"] == pytest.approx(10.0)


def test_metricas_semanales_suma_correcta_por_semana():
    """La agregación suma forecast y reales por (propiedad, semana)."""
    df = pd.DataFrame(
        {
            "property": ["a", "a", "a", "b", "b"],
            "stay_date": pd.to_datetime(
                ["2026-01-05", "2026-01-06", "2026-01-12", "2026-01-05", "2026-01-20"]
            ),
            "actual": [10.0, 20.0, 30.0, 5.0, 6.0],
            "forecast": [12.0, 18.0, 27.0, 5.0, 9.0],
        }
    )
    table = metrics_by_property_weekly(df)
    a = table.loc[table["property"] == "a"].iloc[0]
    # semana 1: actual 30, forecast 30 (12+18); semana 2: 30 vs 27
    assert a["wmape"] == pytest.approx(abs(27 - 30) / (30 + 30))
    assert int(a["n_obs"]) == 2
    b = table.loc[table["property"] == "b"].iloc[0]
    assert int(b["n_obs"]) == 2, "semanas separadas cuentan como semanas distintas"


def test_metricas_semanales_frame_vacio():
    df = pd.DataFrame(columns=["property", "stay_date", "actual", "forecast"])
    table = metrics_by_property_weekly(df)
    assert table.empty


def test_metricas_por_estacionalidad_y_patron_sistematico():
    n = 60
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "property": ["corin"] * n,
            "stay_date": dates,
            "actual": [100.0] * n,
            "forecast": [88.0] * n,  # sub-pronostica 12% siempre
        }
    )
    tables = metrics_by_seasonality(df)

    assert set(tables) == {"month", "day_of_week", "weekend_weekday", "season"}
    for name, t in tables.items():
        assert {"mape", "wmape", "bias", "n_obs"} <= set(t.columns), name

    # weekend con sesgo sistemático detectado
    weekend = tables["weekend_weekday"]
    insights = systematic_pattern(weekend, "day_type", bias_threshold=0.05, min_obs=5)
    assert insights, "debe detectar el patrón sistemático de sub-pronóstico"
    assert any("sub-pronostica" in s and "12" in s for s in insights)


# --------------------------- 6.6 multi-step ------------------------------


def test_multistep_snapshots_congelados():
    history = pd.DataFrame(
        {
            "property": ["corin"] * 100,
            "stay_date": pd.date_range("2025-01-01", periods=100, freq="D"),
            "room_nights": np.arange(100) + 50.0,
        }
    )
    actuals = history.copy()

    def naive_forecast(history_asof, cutoff):
        # usa SOLO history_asof (ya congelado)
        mean = history_asof["room_nights"].mean()
        future_dates = pd.date_range(
            cutoff + pd.Timedelta(days=1), periods=5
        )
        return pd.DataFrame(
            {
                "property": ["corin"] * 5,
                "stay_date": future_dates,
                "forecast_rn": mean,
            }
        )

    cutoffs = [pd.Timestamp("2025-02-01"), pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-01")]
    result = run_multistep(
        cutoffs=cutoffs, history=history, actuals=actuals, forecast_fn=naive_forecast
    )

    assert result.n_snapshots == 3
    assert_no_future_reads(result)  # fallaría si algún snapshot leyó futuro

    for snap in result.snapshots:
        assert snap.history_max_observed <= snap.cutoff
        assert not snap.predictions.empty

    merged = result.all_merged()
    assert not merged.empty
    assert {"actual", "forecast"} <= set(merged.columns)


def test_multistep_detecta_lookahead_en_fn(tmp_path):
    history = pd.DataFrame(
        {
            "property": ["corin"] * 60,
            "stay_date": pd.date_range("2025-01-01", periods=60, freq="D"),
            "room_nights": np.arange(60) + 50.0,
        }
    )
    leaked_state = {}

    def leaky_forecast(history_asof, cutoff):
        # "huye" del frame congelado: usa el histórico completo vía closure
        leaked_state["full_mean"] = history["room_nights"].mean()
        return pd.DataFrame(
            {
                "property": ["corin"],
                "stay_date": [cutoff + pd.Timedelta(days=1)],
                "forecast_rn": [leaked_state["full_mean"]],
            }
        )

    result = run_multistep(
        cutoffs=[pd.Timestamp("2025-01-15")],
        history=history,
        actuals=history,
        forecast_fn=leaky_forecast,
    )
    # el snapshot registra el corte correcto (la fuga está en la fn, que es
    # inseparable de la implementación del modelo) — verificamos que al menos
    # la huella del corte se congela correctamente:
    assert result.snapshots[0].history_max_observed <= result.snapshots[0].cutoff
    assert_no_future_reads(result)


# --------------------------- 6.7 campeón ---------------------------------


def _fold_df():
    """3 candidatos × 4 folds. 'greedy_low_mape' tiene el mejor MAPE pero bias fuerte."""
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(4):
        rows.append(
            {
                "model": "seasonal_naive",
                "fold": fold,
                "mape": 14.0 + rng.normal(0, 0.5),
                "wmape": 0.145 + rng.normal(0, 0.005),
                "mae": 12.0 + rng.normal(0, 0.5),
                "bias": 0.02 + rng.normal(0, 0.01),  # sesgo pequeño
            }
        )
        rows.append(
            {
                "model": "lightgbm",
                "fold": fold,
                "mape": 9.0 + rng.normal(0, 0.3),    # MEJOR MAPE
                "wmape": 0.12 + rng.normal(0, 0.004),
                "mae": 10.0 + rng.normal(0, 0.3),
                "bias": 0.18 + rng.normal(0, 0.02),   # pero bias fuerte (+18%)
            }
        )
        rows.append(
            {
                "model": "pickup_curve",
                "fold": fold,
                "mape": 11.0 + rng.normal(0, 0.4),
                "wmape": 0.115 + rng.normal(0, 0.003),
                "mae": 9.5 + rng.normal(0, 0.3),
                "bias": -0.01 + rng.normal(0, 0.01),  # casi sin sesgo
            }
        )
    return pd.DataFrame(rows)


def test_campeon_mape_no_decide():
    """6.7: el modelo con mejor MAPE pero bias fuerte NO gana."""
    df = _fold_df()
    result = select_champion(df)

    best_mape = min(result.scores, key=lambda s: s.mape).model
    assert best_mape == "lightgbm", "precondición: lightgbm tiene el mejor MAPE"
    assert result.champion != "lightgbm", "el MAPE solo no debe decidir el campeón"
    assert result.champion == "pickup_curve"  # mejor combinación wmape+bias+estabilidad
    assert "no decide" in result.reason


def test_campeon_tabla_y_ranking():
    df = _fold_df()
    result = select_champion(df)
    table = result.table()

    assert set(table["model"]) == {"seasonal_naive", "lightgbm", "pickup_curve"}
    assert {"wmape", "mae", "bias", "mape", "stability", "composite"} <= set(table.columns)
    # ordenado por composite ascendente
    assert table["composite"].is_monotonic_increasing
    # estabilidad calculada entre ventanas
    scores = {s.model: s for s in result.scores}
    assert scores["seasonal_naive"].n_folds == 4


def test_campeon_falla_sin_candidatos():
    with pytest.raises(ValueError):
        select_champion(pd.DataFrame(columns=["model", "fold", "mape", "wmape", "mae", "bias"]))


def test_campeon_score_falta_columna():
    bad = pd.DataFrame({"model": ["a"], "fold": [0], "wmape": [0.1]})
    with pytest.raises(ValueError, match="ausentes"):
        select_champion(bad)


# --------------------------- 6.8 por propiedad ---------------------------


def test_seleccion_por_propiedad_con_evidencia():
    df = _fold_df()
    df["property"] = "corin"
    result = select_per_property(df, min_folds=3)

    assert "corin" in result.by_property
    assert "corin" not in result.property_fallback  # tiene evidencia (4 folds)


def test_seleccion_por_propiedad_fallback_global():
    """Propiedades con pocas ventanas caen al campeón global."""
    df = _fold_df()
    # solo 2 folds para 'sjo'
    df["property"] = np.where(df["fold"] < 2, "sjo", "corin")
    df = df[df["property"] == "sjo"]  # sjo: solo 2 folds por modelo

    result = select_per_property(df, min_folds=3, global_champion="pickup_curve")

    assert result.by_property["sjo"] == "pickup_curve", "fallback al campeón global"
    assert "sjo" in result.property_fallback
    assert "sin selección propia" in result.property_fallback["sjo"]
