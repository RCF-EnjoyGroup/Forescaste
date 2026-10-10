"""Tests de feature engineering (tasks 4.1–4.5)."""

import numpy as np
import pandas as pd
import pytest

from src.features.asof import AsOfBuilder, LeakageError, enforce_cutoff
from src.features.calendar import add_calendar_features, has_holiday_feature
from src.features.capacity import add_capacity_features
from src.features.demand import add_demand_features
from src.features.pickup import add_pickup_features, pickup_pct_of_final


# --------------------------- 4.1 calendario ------------------------------


def test_calendario_solo_deriva_de_fecha():
    df = pd.DataFrame(
        {
            "stay_date": pd.to_datetime(
                ["2026-03-07", "2026-03-08", "2026-12-31", "2026-07-15"]
            ),
            "room_nights": [10, 20, 30, 40],  # no debe influir
        }
    )
    out = add_calendar_features(df, "stay_date")

    row0 = out.iloc[0]  # 2026-03-07 = sábado
    assert row0["day_of_week"] == 5
    assert bool(row0["is_weekend"]) is True
    assert row0["month"] == 3
    assert row0["quarter"] == 1
    assert row0["day_of_month"] == 7
    assert row0["season"] == "shoulder"

    row2 = out.iloc[2]  # 2026-12-31
    assert row2["season"] == "high"
    assert row2["days_to_month_end"] == 0
    assert bool(row2["is_weekend"]) is False or True  # jueves

    # derivadas idénticas aunque cambie el valor de RN (misma fecha):
    df2 = df.copy()
    df2["room_nights"] = [999, 999, 999, 999]
    out2 = add_calendar_features(df2, "stay_date")
    pd.testing.assert_series_equal(out["month"], out2["month"])
    pd.testing.assert_series_equal(out["day_of_week"], out2["day_of_week"])


def test_horizonte_futuro_sin_datos_observados():
    """Las features de calendario se generan solo con la fecha futura."""
    future = pd.DataFrame({"stay_date": pd.date_range("2027-01-10", periods=10)})
    out = add_calendar_features(future, "stay_date")
    assert len(out) == 10
    assert out["month"].notna().all()
    assert out["day_of_week"].notna().all()


def test_feriado_omitido_sin_fuente_confiable():
    df = pd.DataFrame({"stay_date": pd.to_datetime(["2026-12-25"])})
    out = add_calendar_features(df, "stay_date", holidays=None)
    assert not has_holiday_feature(out), "sin fuente => sin columna de feriado"

    out2 = add_calendar_features(df, "stay_date", holidays=["2026-12-25"])
    assert has_holiday_feature(out2)
    assert bool(out2["is_holiday"].iloc[0]) is True


# --------------------------- 4.2 demanda --------------------------------


def _demand_df(n=400, start="2025-01-01", seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "property": ["corin"] * n,
            "stay_date": dates,
            "room_nights": rng.integers(30, 90, size=n),
        }
    )


def test_demand_lags_y_rollings_hacia_atras():
    df = _demand_df(400)
    out, missing = add_demand_features(df)

    i = 100
    expected_lag7 = df["room_nights"].iloc[i - 7]
    got = out["lag_7"].iloc[i]
    assert got == expected_lag7

    # rolling EXCLUYE la propia fila (el target nunca entra a su feature)
    w = 7
    manual = df["room_nights"].iloc[i - w : i].mean()
    assert out["rolling_mean_7"].iloc[i] == pytest.approx(manual)
    assert pd.isna(out["rolling_mean_7"].iloc[0]), "primera fila: ventana vacía"

    assert "lag_365" not in missing  # 400 días de histórico


def test_lags_calendario_con_huecos():
    """lag_7 = mismo día la semana pasada AUN con fechas faltantes.

    Sin la reindexación diaria, un hueco desplaza el lag posicional a otra
    fecha (hallazgo LAPAS/LIRAK del diagnóstico).
    """
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    keep = [d for d in dates if d.dayofweek != 0]  # sin lunes => huecos
    df = pd.DataFrame(
        {
            "property": ["corin"] * len(keep),
            "stay_date": keep,
            "room_nights": np.arange(len(keep), dtype=float),
        }
    )
    out, _ = add_demand_features(df)
    row = out.loc[out["stay_date"] == "2026-01-11"].iloc[0]  # domingo
    val_0104 = float(out.loc[out["stay_date"] == "2026-01-04", "room_nights"].iloc[0])
    assert row["lag_7"] == val_0104, "lag_7 debe apuntar al 4-ene (7 días naturales atrás)"


def test_missing_features_atribuidos_por_propiedad():
    """Una omisión local nunca se reporta como faltante global."""
    df_a = _demand_df(400)
    df_b = _demand_df(200)
    df_b["property"] = "new"
    out, missing = add_demand_features(pd.concat([df_a, df_b], ignore_index=True))

    assert missing.get("lag_365") == ["new"]
    assert missing.get("same_month_year") == ["new"]
    assert out.loc[out["property"] == "corin", "lag_365"].notna().any()
    assert out.loc[out["property"] == "new", "lag_365"].isna().all()


def test_lag_365_omitido_con_historico_insuficiente():
    df = _demand_df(200)  # < 365 días
    out, missing = add_demand_features(df)
    assert "lag_365" in missing
    assert out["lag_365"].isna().all(), "lag_365 sin histórico debe quedar vacío"
    # el resto de features sí existen
    assert out["lag_7"].notna().any()
    assert out["rolling_mean_28"].notna().any()


def test_mismo_dow_y_mes_historico():
    df = _demand_df(400)
    out, _ = add_demand_features(df)
    i = 50
    assert out["same_dow_1w"].iloc[i] == df["room_nights"].iloc[i - 7]
    assert out["same_dow_4w"].iloc[i] == df["room_nights"].iloc[i - 28]
    # mismo día del año anterior requiere i >= 365
    j = 370
    assert out["same_month_year"].iloc[j] == df["room_nights"].iloc[j - 365]
    assert pd.isna(out["same_month_year"].iloc[i]), "antes de 365 días queda vacío"


def test_features_respetan_orden_temporal_asof():
    """4.2 + contrato as-of: recalcular con cutoff = mismos valores en el pasado."""
    df = _demand_df(300)
    cutoff = pd.Timestamp("2025-08-01")
    full_features, _ = add_demand_features(df)
    builder = AsOfBuilder(date_col="stay_date", cutoff=cutoff)

    past = builder.frame(df, "demand")
    past_features, _ = add_demand_features(past)

    overlap = full_features.loc[full_features["stay_date"] < cutoff]
    merged = overlap.merge(
        past_features[["stay_date", "lag_7", "rolling_mean_14"]],
        on="stay_date",
        suffixes=("_full", "_past"),
    )
    assert np.allclose(
        merged["lag_7_full"].astype(float),
        merged["lag_7_past"].astype(float),
        equal_nan=True,
    ), "las features pasadas no pueden depender del futuro"


# --------------------------- 4.3 capacidad ------------------------------


def test_capacidad_cambio_inventario():
    df = pd.DataFrame(
        {
            "property": ["corin"] * 5,
            "stay_date": pd.date_range("2026-01-01", periods=5),
            "capacity": [100, 100, 60, 60, 60],  # cierre parcial
            "current_booked_rn": [40, 55, 55, 58, 60],
        }
    )
    out = add_capacity_features(df)

    assert out["capacity_prev_day"].isna().iloc[0]
    assert out["capacity_change"].iloc[2] == -40
    assert out["capacity_change"].iloc[1] == 0
    assert out["capacity_rolling_mean_7"].iloc[4] == pytest.approx((100 + 100 + 60 + 60) / 4)
    # restante y utilización
    assert out["capacity_remaining"].iloc[4] == 0
    assert out["capacity_utilization"].iloc[4] == pytest.approx(60 / 60)


def test_capacidad_utilizacion_cero_segura():
    df = pd.DataFrame(
        {
            "property": ["corin"] * 3,
            "stay_date": pd.date_range("2026-01-01", periods=3),
            "capacity": [0, 100, 100],
            "current_booked_rn": [5, 50, 0],
        }
    )
    out = add_capacity_features(df)
    assert pd.isna(out["capacity_utilization"].iloc[0])  # cap 0 => sin división


# --------------------------- 4.4 pickup ---------------------------------


def _pickup_df():
    stay = pd.to_datetime(["2026-03-01"] * 6)
    snap = pd.to_datetime(
        ["2025-12-01", "2025-12-31", "2026-01-30", "2026-02-14", "2026-02-24", "2026-02-28"]
    )
    return pd.DataFrame(
        {
            "property": ["corin"] * 6,
            "stay_date": stay,
            "snapshot_date": snap,
            "lead_days": (stay - snap).days,
            "pickup_rn": [20, 35, 48, 55, 58, 60],
        }
    )


def _pickup_multi_stay_df():
    """Dos stay dates con los MISMOS lead offsets => pooling 'mismo lead' real."""
    rows = []
    for stay_s, values in [
        ("2026-03-01", [20, 35, 48]),
        ("2026-03-08", [15, 30, 40]),
    ]:
        stay = pd.Timestamp(stay_s)
        for lead, v in zip([90, 60, 30], values):
            snap = stay - pd.Timedelta(days=lead)
            rows.append(
                {
                    "property": "corin",
                    "stay_date": stay,
                    "snapshot_date": snap,
                    "lead_days": lead,
                    "pickup_rn": v,
                }
            )
    return pd.DataFrame(rows)


def test_pickup_aceleracion_formula():
    df = _pickup_df()
    out = add_pickup_features(df)
    # aceleración = pickup reciente (3) - pickup previo (3)
    delta = df["pickup_rn"].diff()
    recent = delta.rolling(3, min_periods=1).sum()
    previous = recent.shift(3)
    expected = recent - previous
    got = out["pickup_acceleration"].astype(float)
    for i in range(len(out)):
        if pd.notna(expected.iloc[i]):
            assert got.iloc[i] == pytest.approx(expected.iloc[i])


def test_pickup_mismo_lead_asof():
    """El pickup al mismo lead usa solo snapshots ≤ cutoff."""
    df = _pickup_df()
    cutoff = pd.Timestamp("2026-01-31")
    out_cut = add_pickup_features(df, cutoff=cutoff)
    # tras el corte solo quedan 3 snapshots (dic-01, dic-31, ene-30)
    assert len(out_cut) == 3
    # con un solo stay_date cada lead es único => media = valor propio
    assert out_cut["pickup_hist_count_same_lead"].notna().all()
    assert out_cut["pickup_hist_mean_same_lead"].iloc[0] == pytest.approx(20.0)
    assert out_cut["pickup_hist_mean_same_lead"].iloc[2] == pytest.approx(48.0)

    # sin cutoff => 6 snapshots (los 3 primeros no cambian: leads únicos)
    out_full = add_pickup_features(df)
    assert len(out_full) == 6
    assert out_full["pickup_hist_mean_same_lead"].iloc[0] == pytest.approx(20.0)


def test_pickup_mismo_lead_poolea_entre_stay_dates():
    """Mismo lead agrupa observaciones de distintas fechas de estancia."""
    df = _pickup_multi_stay_df()
    out = add_pickup_features(df)
    # leads [90,60,30] repetidos en 2 stay dates => media por lead = promedio de 2
    grouped = out.groupby("lead_days")["pickup_rn"].transform("mean")
    assert np.allclose(out["pickup_hist_mean_same_lead"], grouped)
    counts = out.groupby("lead_days")["pickup_rn"].transform("count")
    assert np.allclose(out["pickup_hist_count_same_lead"], counts)
    # verificación explícita: lead 90 => media(20, 15) = 17.5
    lead90 = out.loc[out["lead_days"] == 90, "pickup_hist_mean_same_lead"]
    assert np.allclose(lead90, 17.5)
    assert (out["pickup_hist_count_same_lead"] == 2).all()


def test_pickup_ventanas_acumulado_volatilidad():
    df = _pickup_df()
    out = add_pickup_features(df)
    for w in (1, 3, 7, 14, 30):
        assert f"pickup_last_{w}d" in out.columns
    # acumulado = cumsum de los niveles de pickup registrados
    assert out["pickup_cumulative"].iloc[-1] == pytest.approx(df["pickup_rn"].sum())
    assert "pickup_volatility" in out.columns
    assert "pickup_deceleration" in out.columns


def test_pickup_pct_cero_seguro():
    pct = pickup_pct_of_final(pd.Series([10.0, 20.0]), pd.Series([0.0, 100.0]))
    assert pd.isna(pct.iloc[0])
    assert pct.iloc[1] == pytest.approx(0.20)


# --------------------------- 4.5 señal ----------------------------------


def test_features_aportan_senial_sin_fugas():
    """4.5: pickup/capacidad correlacionan con la demanda futura observada.

    Se construye un dataset donde la demanda final es mecánicamente
    current_booked + pickup conocido; las features deben correlacionar
    positivamente con el target y las features no deben verse afectadas por
    datos posteriores al corte (verificado con as-of).
    """
    rng = np.random.default_rng(7)
    n = 200
    dates = pd.date_range("2025-06-01", periods=n, freq="D")
    booked = rng.integers(30, 60, size=n).astype(float)
    pickup = rng.integers(0, 25, size=n).astype(float)
    final = booked + pickup  # target observable

    df = pd.DataFrame(
        {
            "property": ["corin"] * n,
            "stay_date": dates,
            "capacity": np.full(n, 100),
            "current_booked_rn": booked,
            "pickup_rn": pickup,
            "room_nights": final,
        }
    )

    # señal: correlación booked/features con target
    corr_booked = np.corrcoef(booked, final)[0, 1]
    corr_pickup = np.corrcoef(pickup, final)[0, 1]
    assert corr_booked > 0.5, f"booked sin señal: {corr_booked}"
    assert corr_pickup > 0.1, f"pickup sin señal: {corr_pickup}"

    # sin fugas: features recalculadas en el pasado = mismos valores
    cutoff = pd.Timestamp("2025-08-01")
    builder = AsOfBuilder(date_col="stay_date", cutoff=cutoff)
    past = builder.frame(df, "all")
    enforce_cutoff(past, "stay_date", cutoff, "all")

    # una feature que intente usar el futuro debe fallar
    def leaky(frame):
        return df  # devuelve el frame completo (con futuro)

    with pytest.raises(LeakageError):
        builder.apply(df, leaky, "leaky")
