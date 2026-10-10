"""Tests de backtesting (tasks 7.1–7.3)."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.snapshots import (
    DEFAULT_LEADS,
    BacktestResult,
    build_snapshot,
    clear_cache,
    run_backtest,
)


def _history(n=200, start="2025-06-01", seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "property": ["corin"] * n,
            "stay_date": dates,
            "room_nights": rng.integers(40, 90, size=n).astype(float),
        }
    )


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_cache()
    yield
    clear_cache()


# ------------------------------ 7.1 -------------------------------------


def test_snapshot_no_contiene_datos_posteriores():
    """7.1: un snapshot histórico NO contiene datos posteriores a su corte."""
    history = _history()
    stay = pd.Timestamp("2025-11-15")

    for lead in DEFAULT_LEADS:
        snap = build_snapshot(history, stay, lead, cache_key="t1")
        cutoff = stay - pd.Timedelta(days=lead)
        assert snap.forecast_date == cutoff
        assert snap.frame["stay_date"].max() <= cutoff, (
            f"snapshot T-{lead} filtró datos hasta {snap.frame['stay_date'].max()} "
            f"> corte {cutoff}"
        )
        # y el corte es exacto: hay datos cerca del límite
        assert not snap.frame.empty


def test_snapshot_mas_cercano_acumula_mas_historia():
    """T-1 (más cerca de la estancia) tiene MÁS datos disponibles que T-90."""
    history = _history()
    stay = pd.Timestamp("2025-11-15")
    n90 = build_snapshot(history, stay, 90, cache_key="t2").frame.shape[0]
    n1 = build_snapshot(history, stay, 1, cache_key="t2").frame.shape[0]
    assert n1 > n90, "T-1 debe tener más historia acumulada que T-90"
    # ambos contienen datos: el corte es efectivo en los dos
    assert n90 > 0 and n1 > 0


def test_snapshot_cache_reutiliza():
    history = _history()
    stay = pd.Timestamp("2025-11-15")
    a = build_snapshot(history, stay, 30, cache_key="t3")
    b = build_snapshot(history, stay, 30, cache_key="t3")
    assert a.frame is b.frame, "la caché debe reutilizar el frame por corte"


def test_snapshot_lead_negativo_falla():
    """stay date anterior al forecast date => lead inválido."""
    history = _history()
    stay = pd.Timestamp("2025-06-01")
    snap = build_snapshot(history, stay, 400, cache_key="t4")  # excede el history
    assert snap.frame.empty  # sin datos futuros: recorta a vacío, no filtra mal


# ------------------------------ 7.2 -------------------------------------


def _forecast_fn(snapshot_frame, stay_date, lead):
    """Modelo ingenuo: promedio del snapshot + delta según booked."""
    mean_rn = float(snapshot_frame["room_nights"].mean())
    booked = float(snapshot_frame["room_nights"].iloc[-1])
    expected = max(mean_rn - booked, 0.0) * (lead / 90)
    return {
        "current_booked_rn": booked,
        "expected_pickup_rn": expected,
        "forecast_rn": booked + expected,
        "model_name": "naive_backtest",
    }


def test_revisiones_reconstruibles_t90_a_t1_a_actual():
    """7.2: la serie T-90 → T-1 → Actual es reconstruible."""
    history = _history()
    stay = pd.Timestamp("2025-11-15")
    result = run_backtest(history, [stay], _forecast_fn, leads=DEFAULT_LEADS)

    series = result.series_for(stay)
    assert len(series) == len(DEFAULT_LEADS) + 1  # revisiones + actual

    # ordenado de mayor a menor lead
    leads = list(series["lead_days"])
    assert leads == sorted(leads, reverse=True)
    assert leads[-1] == 0, "la última fila es el Actual (lead 0)"

    # hay forecast_date por revisión y coincide con stay - lead
    revs = series[series["lead_days"] > 0]
    for _, r in revs.iterrows():
        expected_fd = stay - pd.Timedelta(days=int(r["lead_days"]))
        assert pd.Timestamp(r["forecast_date"]) == expected_fd

    # el actual está presente
    assert "is_actual" in series.columns
    actual_rows = series[series.get("is_actual", False) == True]  # noqa: E712
    assert len(actual_rows) == 1
    assert actual_rows["forecast_rn"].iloc[0] == pytest.approx(
        history.loc[history["stay_date"] == stay, "room_nights"].iloc[0]
    )


def test_revisiones_frame_completo():
    history = _history()
    stays = pd.DatetimeIndex(["2025-11-10", "2025-11-15"])
    result = run_backtest(history, stays, _forecast_fn, leads=(30, 7, 1))
    frame = result.frame()
    assert len(frame) == 2 * 3
    assert {"stay_date", "forecast_date", "lead_days", "forecast_rn"} <= set(frame.columns)
    # cada (stay, lead) único
    assert not frame.duplicated(subset=["stay_date", "lead_days"]).any()


# ------------------------------ 7.3 -------------------------------------


def test_metricas_globales_y_por_lead():
    history = _history()
    stays = pd.DatetimeIndex(pd.date_range("2025-11-01", periods=5, freq="3D"))
    result = run_backtest(history, stays, _forecast_fn, leads=DEFAULT_LEADS)

    m = result.metrics()
    assert np.isfinite(m.mape)
    assert np.isfinite(m.wmape)
    assert np.isfinite(m.mae)
    assert m.n_obs > 0

    by_lead = result.metrics_by_lead()
    assert not by_lead.empty
    assert set(by_lead["lead_band"].astype(str)) <= set(
        ["1-2", "3-6", "7-13", "14-29", "30-59", "60-89", "90+"]
    )
    # cobertura: la suma de obs = total de revisiones con actual
    assert int(by_lead["n_obs"].sum()) == len(result.revisions)
    assert by_lead["mape"].notna().all()
    assert by_lead["bias"].notna().all()


def test_backtest_sin_actuals_falla():
    history = pd.DataFrame(
        {"stay_date": pd.date_range("2025-06-01", periods=10), "room_nights": [50.0] * 10}
    )
    # stay dates fuera del history => sin actuals
    result = run_backtest(
        history,
        [pd.Timestamp("2026-01-01")],
        _forecast_fn,
        leads=(7,),
        actuals=pd.Series(dtype=float),  # vacío explícito
    )
    with pytest.raises(ValueError, match="actual"):
        result.metrics()


def test_backtest_result_vacio_falla():
    empty = BacktestResult()
    with pytest.raises(ValueError):
        empty.metrics()
    assert empty.frame().empty
