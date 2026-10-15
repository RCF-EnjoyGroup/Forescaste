"""Tests de modelado (tasks 5.1–5.8)."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.constraints import (
    apply_capacity_constraint,
    build_intervals,
    compute_occupancy,
    occupancy_violations,
    reconcile_portfolio,
)
from src.models.baselines import PickupCurveModel, SeasonalNaive
from src.models.candidates import (
    CatBoostModel,
    ETSModel,
    LightGBMModel,
    SARIMAXModel,
    XGBoostModel,
)


# ----------------------------- 5.1 Seasonal Naive ------------------------


def test_seasonal_naive_produce_por_propiedad_fecha():
    hist = pd.DataFrame(
        {
            "stay_date": pd.date_range("2026-01-05", periods=28, freq="D"),
            "room_nights": np.arange(28) + 100,
        }
    )
    model = SeasonalNaive().fit(hist)
    future = pd.date_range("2026-02-02", periods=7)
    out = model.predict(future)

    assert len(out) == 7
    assert set(out["model_name"]) == {"seasonal_naive"}
    assert out["stay_date"].notna().all()
    assert out["forecast_rn"].notna().all()
    assert (out["forecast_rn"] > 0).all()
    # lunes 2026-02-02 => media de las últimas 4 ocurrencias del mismo DOW:
    # 2026-01-05 (100), 01-12 (107), 01-19 (114), 01-26 (121) → 110.5
    mon = out.loc[out["stay_date"] == pd.Timestamp("2026-02-02"), "forecast_rn"].iloc[0]
    assert mon == pytest.approx(110.5)


def test_seasonal_naive_media_ultimas_k_ocurrencias():
    """El baseline usa la MEDIA de las últimas K ocurrencias del mismo DOW.

    Una sola observación es un contrincante débil: el benchmark entero
    pierde credibilidad si los candidatos 'ganan' contra ruido. Con K=4 el
    ruido se suaviza sin aplanar la estacionalidad semanal.
    """
    # 56 días = 8 semanas → 8 ocurrencias de cada DOW
    hist = pd.DataFrame(
        {
            "stay_date": pd.date_range("2026-01-05", periods=56, freq="D"),
            "room_nights": np.arange(56) + 100,
        }
    )
    mon = pd.Timestamp("2026-03-02")
    # los 8 lunes previos: 2026-01-05..2026-02-23 → valores 100, 107, ..., 149
    for k, expected in [(1, 149.0), (2, 145.5), (4, 138.5), (8, 124.5)]:
        model = SeasonalNaive(dow_window=k).fit(hist)
        out = model.predict(pd.DatetimeIndex([mon]))
        assert out["forecast_rn"].iloc[0] == pytest.approx(expected)


def test_seasonal_naive_multi_propiedad():
    frames = []
    for prop, offset in [("corin", 0), ("marina", 50)]:
        frames.append(
            pd.DataFrame(
                {
                    "property": [prop] * 21,
                    "stay_date": pd.date_range("2026-01-05", periods=21),
                    "room_nights": np.arange(21) + 100 + offset,
                }
            )
        )
    hist = pd.concat(frames, ignore_index=True)
    future = pd.date_range("2026-01-26", periods=3)
    for prop in ["corin", "marina"]:
        model = SeasonalNaive().fit(hist.loc[hist["property"] == prop])
        out = model.predict(future)
        assert len(out) == 3
        assert out["forecast_rn"].notna().all()


# ----------------------------- 5.2 Pickup Curve --------------------------


def _curves_df():
    """Curvas: pickup esperado decrece a medida que baja el lead."""
    rows = []
    rng = np.random.default_rng(0)
    for stay in pd.date_range("2026-03-01", periods=3, freq="7D"):
        for lead in [90, 60, 30, 14, 7, 3, 1]:
            # booking curve: pickup = 40 * (1 - exp(-lead/30)) + ruido
            total = 40 * (1 - np.exp(-lead / 30)) + rng.normal(0, 1.5)
            rows.append(
                {
                    "property": "corin",
                    "stay_date": stay,
                    "lead_days": lead,
                    "pickup_rn": max(total, 0),
                    "current_booked_rn": 50,
                }
            )
    return pd.DataFrame(rows)


def test_pickup_curve_trayectoria_decreciente():
    """5.2: T-90 → T-1 el pickup esperado decrece conforme baja el lead."""
    model = PickupCurveModel(n_bins=7).fit(_curves_df())
    req = pd.DataFrame(
        {
            "property": ["corin"] * 7,
            "stay_date": [pd.Timestamp("2026-03-22")] * 7,
            "lead_days": [90, 60, 30, 14, 7, 3, 1],
            "current_booked_rn": [50.0] * 7,
        }
    )
    out = model.predict(req)
    assert list(out["model_name"]) == ["pickup_curve"] * 7
    pickups = out["expected_pickup_rn"].to_numpy()
    # decreciente (no estricto: bins pueden empatar, pero no puede aumentar)
    diffs = np.diff(pickups)
    assert (diffs <= 1e-9).all(), f"pickup no decreciente: {pickups}"
    # forecast = booked + pickup
    assert np.allclose(out["forecast_rn"], out["current_booked_rn"] + out["expected_pickup_rn"])


def test_pickup_curve_fora_de_rango_usa_extremo():
    model = PickupCurveModel(n_bins=7).fit(_curves_df())
    req = pd.DataFrame(
        {
            "property": ["corin"],
            "stay_date": [pd.Timestamp("2026-03-22")],
            "lead_days": [400],  # fuera de rango
            "current_booked_rn": [50.0],
        }
    )
    out = model.predict(req)
    assert out["forecast_rn"].iloc[0] >= 50
    assert np.isfinite(out["forecast_rn"].iloc[0])


def _short_lead_curves_df():
    """Curvas con gradiente fuerte a leads cortos: lead 1 ≈ 0, lead 9 ≈ 21.

    Con bins UNIFORMES de ~9 días (n_bins=10 sobre lead<=90) el lead 1 y el
    9 comparten bin: la curva comprime el gradiente hacia la media del bin y
    sobre-pronostica al cierre de la reserva.
    """
    rows = []
    rng = np.random.default_rng(7)
    for stay in pd.date_range("2026-03-01", periods=12, freq="7D"):
        for lead in [1, 2, 3, 5, 7, 9, 14, 21, 30, 45, 60, 90]:
            remaining = max(67.0 * (1.0 - np.exp(-lead / 9.0)) + rng.normal(0, 0.5), 0.0)
            rows.append(
                {
                    "property": "corin",
                    "stay_date": stay,
                    "lead_days": lead,
                    "pickup_rn": remaining,
                    "current_booked_rn": 50.0,
                }
            )
    return pd.DataFrame(rows)


def test_bin_edges_densos_resuelven_gradiente_corto():
    """Bordes explícitos: lead 1 y lead 9 NO comparten pickup esperado."""
    req = pd.DataFrame(
        {
            "property": ["corin"] * 2,
            "stay_date": [pd.Timestamp("2026-05-30")] * 2,
            "lead_days": [1, 9],
            "current_booked_rn": [50.0, 50.0],
        }
    )
    # bins uniformes: [1, 10.9) agrupa leads 1..9 → mismo pickup
    uniform = PickupCurveModel(n_bins=10).fit(_short_lead_curves_df())
    pu = uniform.predict(req)["expected_pickup_rn"].to_numpy()
    # bordes densos: [1,3), [3,7), [7,14) separan el gradiente
    dense = PickupCurveModel(bin_edges=[1, 3, 7, 14, 21, 30, 45, 60, 90]).fit(
        _short_lead_curves_df()
    )
    pd_pick = dense.predict(req)["expected_pickup_rn"].to_numpy()
    assert pu[0] == pu[1], "precondición: bins uniformes agrupan lead 1 y 9"
    assert pd_pick[1] > pd_pick[0] + 5.0, (
        f"bins densos deben separar el gradiente: lead1={pd_pick[0]:.1f} "
        f"lead9={pd_pick[1]:.1f}"
    )


def test_bin_edges_catch_all_mas_alla_del_ultimo_borde():
    """Leads ≥ último borde caen en el bin final (catch-all), sin NaN."""
    model = PickupCurveModel(bin_edges=[1, 3, 7, 14, 21, 30]).fit(_short_lead_curves_df())
    req = pd.DataFrame(
        {
            "property": ["corin"] * 3,
            "stay_date": [pd.Timestamp("2026-05-30")] * 3,
            "lead_days": [45, 90, 400],
            "current_booked_rn": [50.0, 50.0, 50.0],
        }
    )
    out = model.predict(req)
    assert out["expected_pickup_rn"].notna().all()
    # los tres leads largos comparten el bin catch-all → mismo pickup
    assert out["expected_pickup_rn"].nunique() == 1


def test_bin_edges_validacion_estrictamente_crecientes():
    with pytest.raises(ValueError, match="estrictamente crecientes"):
        PickupCurveModel(bin_edges=[1, 5, 5, 9]).fit(_curves_df())
    with pytest.raises(ValueError, match="estrictamente crecientes"):
        PickupCurveModel(bin_edges=[7, 3]).fit(_curves_df())
    with pytest.raises(ValueError, match="min_lead"):
        PickupCurveModel(bin_edges=[0, 3, 7]).fit(_curves_df())


# ----------------------------- 5.3 ETS / SARIMAX -------------------------


def _seasonal_series(n=140, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return pd.Series(50 + 10 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 2, n))


def test_ets_ajusta_y_predice():
    y = _seasonal_series()
    model = ETSModel(seasonal_periods=7).fit(y)
    fc = model.predict(14)
    assert len(fc) == 14
    assert np.isfinite(fc).all()
    # nivel razonable: la serie media ~50
    assert 30 < np.mean(fc) < 70


def test_sarimax_ajusta_y_predice():
    y = _seasonal_series(n=84)
    model = SARIMAXModel(order=(1, 0, 0), seasonal_order=(1, 0, 0, 7)).fit(y)
    fc = model.predict(14)
    assert len(fc) == 14
    assert np.isfinite(fc).all()
    lo, hi = model.prediction_interval(14, alpha=0.20)
    assert (lo <= hi).all()


# ----------------------------- 5.4 ML ------------------------------------


def _ml_dataset(n=300, seed=2):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    dow = dates.dayofweek.to_numpy()
    booked = rng.integers(30, 70, size=n).astype(float)
    cap = np.full(n, 100.0)
    target = booked + 10 * (dow >= 5) + rng.normal(0, 3, n)
    return pd.DataFrame(
        {
            "stay_date": dates,
            "property": ["corin"] * n,
            "current_booked_rn": booked,
            "day_of_week": dow,
            "capacity": cap,
            "lag_1": np.roll(target, 1),
            "room_nights": target,
        }
    )


ML_FEATURES = ["current_booked_rn", "day_of_week", "capacity", "lag_1"]


@pytest.mark.parametrize(
    "model_factory",
    [
        LightGBMModel,
        XGBoostModel,
        CatBoostModel,
    ],
    ids=["lightgbm", "xgboost", "catboost"],
)
def test_ml_candidates_entrenan_y_predicen(model_factory):
    df = _ml_dataset()
    train = df.iloc[:240]
    test = df.iloc[240:].copy()

    model = model_factory().fit(train, "room_nights", ML_FEATURES)
    pred = model.predict(test)

    assert len(pred) == len(test)
    assert np.isfinite(pred).all()
    # señal razonable (MAPE < 50% en un target sintético ruidoso pero claro)
    mape = np.mean(np.abs(pred - test["room_nights"].to_numpy()) / test["room_nights"].to_numpy())
    assert mape < 0.30, f"{model_factory.__name__} sin señal: MAPE={mape:.3f}"


def test_ml_falla_si_falta_feature():
    df = _ml_dataset(n=100)
    with pytest.raises(KeyError):
        LightGBMModel().fit(df, "room_nights", ["no_existe_col"])


# ----------------------------- 5.5 Capacidad -----------------------------


def test_capacidad_nunca_excede_y_se_conserva_raw():
    df = pd.DataFrame(
        {
            "property": ["corin", "corin", "corin"],
            "stay_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "raw_forecast_rn": [120.0, 80.0, 100.0],
            "capacity": [100.0, 100.0, 90.0],
        }
    )
    out, alerts = apply_capacity_constraint(df)

    # nunca excede
    assert (out["final_forecast_rn"] <= out["capacity"] + 1e-9).all()
    # raw se conserva intacto
    assert list(out["raw_forecast_rn"]) == [120.0, 80.0, 100.0]
    assert list(out["capacity"]) == [100.0, 100.0, 90.0]
    assert list(out["final_forecast_rn"]) == [100.0, 80.0, 90.0]

    # alerta emitida ANTES del cap: 2 violaciones (120>100, 100>90)
    assert len(alerts) == 2
    a0 = alerts[0]
    assert a0.raw_forecast == 120.0 and a0.capacity == 100.0 and a0.capped_by == 20.0
    assert a0.property == "corin" and a0.stay_date == "2026-01-01"


def test_capacidad_sin_violaciones_no_alerta():
    df = pd.DataFrame(
        {
            "property": ["corin"],
            "stay_date": pd.to_datetime(["2026-01-01"]),
            "raw_forecast_rn": [50.0],
            "capacity": [100.0],
        }
    )
    out, alerts = apply_capacity_constraint(df)
    assert alerts == []
    assert bool(out["capacity_constraint_applied"].iloc[0]) is False


def test_capacidad_falla_sin_columnas():
    with pytest.raises(KeyError):
        apply_capacity_constraint(pd.DataFrame({"property": ["x"]}))
    with pytest.raises(KeyError):
        apply_capacity_constraint(pd.DataFrame({"raw_forecast_rn": [1.0]}))


# ----------------------------- 5.6 Ocupación -----------------------------


def test_ocupacion_en_rango():
    df = pd.DataFrame(
        {
            "property": ["corin"] * 3,
            "stay_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "final_forecast_rn": [50.0, 100.0, 0.0],
            "capacity": [100.0, 100.0, 100.0],
        }
    )
    out = compute_occupancy(df)
    assert list(out["forecast_occupancy"]) == [0.5, 1.0, 0.0]
    assert occupancy_violations(out).empty
    assert not out["occupancy_out_of_range"].any()


def test_ocupacion_fuera_de_rango_detectada():
    df = pd.DataFrame(
        {
            "property": ["corin"],
            "stay_date": pd.to_datetime(["2026-01-01"]),
            "final_forecast_rn": [150.0],  # > capacidad sin cap
            "capacity": [100.0],
        }
    )
    out = compute_occupancy(df)
    assert out["forecast_occupancy"].iloc[0] == pytest.approx(1.5)
    vio = occupancy_violations(out)
    assert len(vio) == 1


def test_ocupacion_capacidad_cero():
    df = pd.DataFrame(
        {
            "property": ["corin"],
            "stay_date": pd.to_datetime(["2026-01-01"]),
            "final_forecast_rn": [0.0],
            "capacity": [0.0],
        }
    )
    out = compute_occupancy(df)
    assert pd.isna(out["forecast_occupancy"].iloc[0])  # sin división por cero


# ----------------------------- 5.7 Jerarquía -----------------------------


def test_jerarquia_suma_vs_portafolio():
    df = pd.DataFrame(
        {
            "property": ["corin", "corin", "marina", "marina"],
            "stay_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"]),
            "final_forecast_rn": [50.0, 60.0, 40.0, 45.0],
        }
    )
    # portafolio independiente con brecha intencional
    port = pd.Series(
        [95.0, 110.0],
        index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
    )
    result = reconcile_portfolio(df, portfolio_forecast=port)

    # suma por fecha
    assert list(result.property_sum["property_sum"]) == [90.0, 105.0]
    # brecha reportada (95-90=5, 110-105=5)
    assert list(result.gap_by_date["gap"]) == [5.0, 5.0]
    assert result.total_gap == pytest.approx(10.0)
    assert result.gap_pct == pytest.approx(10 / 195)


def test_jerarquia_bottom_up_sin_portafolio_independiente():
    df = pd.DataFrame(
        {
            "property": ["corin", "marina"],
            "stay_date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "final_forecast_rn": [50.0, 40.0],
        }
    )
    result = reconcile_portfolio(df)
    assert result.total_gap == 0.0
    assert result.portfolio["forecast_portfolio"].iloc[0] == pytest.approx(90.0)


# ----------------------------- 5.8 Intervalos ----------------------------


def test_intervalos_orden_p10_p50_p90():
    df = pd.DataFrame(
        {
            "property": ["corin"] * 5,
            "stay_date": pd.date_range("2026-01-01", periods=5),
            "raw_forecast_rn": [50.0, 60.0, 70.0, 80.0, 90.0],
        }
    )
    residuals = pd.Series(np.array([-10.0, -5.0, 0.0, 3.0, 8.0, 12.0]))
    result = build_intervals(df, model_name="lightgbm", residuals=residuals)

    assert result.has_intervals
    result.validate_ordering()  # p10 <= p50 <= p90
    f = result.frame
    assert (f["p10_forecast"] <= f["p50_forecast"]).all()
    assert (f["p50_forecast"] <= f["p90_forecast"]).all()
    assert f["interval_available"].all()
    # banda asimétrica alrededor del punto
    assert (f["p90_forecast"] - f["raw_forecast_rn"]).iloc[0] > 0
    assert (f["p10_forecast"] - f["raw_forecast_rn"]).iloc[0] < 0


def test_modelo_sin_intervalo_marcado():
    df = pd.DataFrame(
        {
            "property": ["corin"],
            "stay_date": [pd.Timestamp("2026-01-01")],
            "raw_forecast_rn": [50.0],
        }
    )
    result = build_intervals(df, model_name="custom_sin_bandas", residuals=None)
    assert not result.has_intervals
    assert "sin intervalos" in result.note
    assert not result.frame["interval_available"].any()
    assert result.frame["p50_forecast"].isna().all()
    result.validate_ordering()  # no falla para modelos sin intervalo


def test_intervalos_orden_invalido_falla():
    df = pd.DataFrame(
        {
            "property": ["corin"],
            "stay_date": [pd.Timestamp("2026-01-01")],
            "raw_forecast_rn": [50.0],
        }
    )
    result = build_intervals(df, model_name="m", residuals=pd.Series([1.0, 2.0]))
    # forzar orden inválido
    result.frame["p10_forecast"] = 999.0
    with pytest.raises(ValueError, match="percentiles"):
        result.validate_ordering()
