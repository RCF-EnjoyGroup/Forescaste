"""Tests anti-leakage (tasks 3.1–3.4).

3.1 as-of builder: para un corte T ninguna feature proviene de datos > T.
3.2 lags/rollings/expansivos solo hacia atrás.
3.3 veto de splits aleatorios.
3.4 inyectar leakage deliberado hace fallar la corrida.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.asof import (
    AsOfBuilder,
    LeakageError,
    enforce_cutoff,
    expanding_backward,
    lag,
    rolling_backward,
)
from src.validation.leakage import (
    assert_as_of,
    check_rolling_with_future,
    run_leakage_suite,
    validate_split_config,
    validate_temporal_order,
)


def _history(periods=60, end="2026-06-30", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=end, periods=periods, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "rn": rng.integers(20, 80, size=periods),
            "capacity": np.full(periods, 100),
            "pickup": rng.integers(0, 30, size=periods),
        }
    )


# ----------------------------- 3.1 --------------------------------------


def test_asof_builder_excluye_futuro():
    df = _history()  # 60 días terminando 2026-06-30 => inicia 2026-05-02
    cutoff = pd.Timestamp("2026-05-31")
    builder = AsOfBuilder(date_col="date", cutoff=cutoff)

    out = builder.frame(df, "demo")
    assert out["date"].max() <= cutoff
    # 05-02..05-31 inclusive = 30 días
    assert len(out) == 30
    assert out["date"].min() == pd.Timestamp("2026-05-02")
    enforce_cutoff(out, "date", cutoff, "demo")  # no lanza


def test_asof_builder_features_no_miran_futuro():
    df = _history()
    cutoff = pd.Timestamp("2026-05-31")
    builder = AsOfBuilder(date_col="date", cutoff=cutoff)

    def features(frame):
        out = frame.copy()
        out["mean_rn"] = out["rn"].mean()
        out["future_marker"] = 0
        return out

    out = builder.apply(df, features, "feat")
    # la media se calculó solo con datos <= corte
    expected = df.loc[df["date"] <= cutoff, "rn"].mean()
    assert out["mean_rn"].iloc[0] == pytest.approx(expected)
    enforce_cutoff(out, "date", cutoff, "feat")


def test_asof_builder_detecta_contaminacion_futura():
    """Si la función ignora el frame recortado y devuelve filas > T, falla."""
    df = _history()
    cutoff = pd.Timestamp("2026-05-31")
    builder = AsOfBuilder(date_col="date", cutoff=cutoff)

    def leaky_features(frame):
        # "olvida" recortar: devuelve el frame completo con fechas futuras
        return df.assign(mean_rn=df["rn"].mean())

    with pytest.raises(LeakageError):
        builder.apply(df, leaky_features, "leaky")


def test_enforce_cutoff_detecta_futuro():
    df = _history()
    with pytest.raises(LeakageError) as exc:
        enforce_cutoff(df, "date", pd.Timestamp("2026-05-31"), "rn")
    assert "futuras" in str(exc.value)
    assert_as_of(df.loc[df["date"] <= "2026-05-31"], "date", pd.Timestamp("2026-05-31"))


def test_targets_futuros_excluidos():
    df = pd.DataFrame({"stay_date": pd.to_datetime(["2026-07-01", "2026-05-01", "2026-06-15"])})
    builder = AsOfBuilder(date_col="stay_date", cutoff=pd.Timestamp("2026-06-01"))
    out = builder.filter_future_targets(df, "stay_date")
    assert list(out["stay_date"].dt.strftime("%Y-%m-%d")) == ["2026-05-01"]


# ----------------------------- 3.2 --------------------------------------


def test_lag_solo_hacia_atras():
    s = pd.Series([1, 2, 3, 4, 5])
    assert lag(s, 1).tolist()[1:] == [1, 2, 3, 4]
    assert np.isnan(lag(s, 1).iloc[0])
    with pytest.raises(ValueError):
        lag(s, -1)  # lag negativo miraría al futuro


def test_rolling_28_excluye_futuro():
    """Task 3.2: media rolling de 28 días no incluye observaciones futuras."""
    rng = np.random.default_rng(1)
    values = rng.integers(0, 100, size=60).astype(float)
    s = pd.Series(values)
    cut_idx = 30  # corte en la posición 30

    r_backward = rolling_backward(s, 28, "mean")
    # referencia manual: media de [i-28..i-1] (la fila propia queda fuera)
    i = cut_idx
    manual = values[i - 28 : i].mean()
    assert r_backward.iloc[i] == pytest.approx(manual)

    # si el rolling mirara al futuro (center=True), el valor en i sería distinto
    centered = s.rolling(window=28, center=True).mean()
    # el valor manual con solo-pasado NO debe ser igual al centrado en general
    future_part = values[i + 1 : i + 28]
    if len(future_part):
        leaky_manual = values[i - 13 : i + 14].mean()  # ventana centrada real
        assert r_backward.iloc[i] != pytest.approx(leaky_manual)


def test_rolling_std_median_y_expanding():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    # la ventana [t-3, t-1] excluye la propia fila
    assert rolling_backward(s, 3, "median").iloc[2] == pytest.approx(1.5)
    assert rolling_backward(s, 3, "std").iloc[2] == pytest.approx(0.7071067811865476)
    assert expanding_backward(s).iloc[3] == pytest.approx(2.5)
    with pytest.raises(ValueError):
        rolling_backward(s, 0, "mean")
    with pytest.raises(ValueError):
        rolling_backward(s, 2, "quantile")


# ----------------------------- 3.3 --------------------------------------


def test_split_aleatorio_prohibido():
    with pytest.raises(LeakageError) as exc:
        validate_split_config({"strategy": "random_split", "test_size": 0.2})
    assert "aleatorio" in str(exc.value) or "temporal" in str(exc.value)


def test_split_shuffle_prohibido():
    with pytest.raises(LeakageError):
        validate_split_config({"strategy": "expanding", "shuffle": True})


def test_split_mixto_falla_antes_de_entrenar():
    with pytest.raises(LeakageError) as exc:
        validate_split_config(
            {"strategy": "temporal", "train_end": "2026-03-31", "val_start": "2026-03-01"}
        )
    assert "mixto" in str(exc.value)


def test_split_expanding_valido():
    validate_split_config(
        {"strategy": "expanding", "train_end": "2026-03-31", "val_start": "2026-04-01"}
    )


def test_orden_temporal_valido_e_invalido():
    train = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=10)})
    val_ok = pd.DataFrame({"date": pd.date_range("2026-01-11", periods=5)})
    validate_temporal_order(train, val_ok, "date")

    val_bad = pd.DataFrame({"date": pd.date_range("2026-01-05", periods=5)})
    with pytest.raises(LeakageError):
        validate_temporal_order(train, val_bad, "date")


# ----------------------------- 3.4 --------------------------------------


def _feature_fn_honest(frame):
    """Features as-of: media acumulada solo hacia atrás."""
    out = frame.copy()
    out["mean_rn"] = out["rn"].expanding().mean()
    return out


def _feature_fn_leaky(frame):
    """Features con leakage: media global del frame que reciba (usa futuro)."""
    out = frame.copy()
    out["mean_rn"] = frame["rn"].mean()
    return out


def test_suite_verde_sin_leakage():
    df = _history()
    cutoff = pd.Timestamp("2026-05-31")
    train = df.loc[df["date"] <= cutoff].copy()
    train["mean_rn"] = train["rn"].expanding().mean()
    full = _feature_fn_honest(df)
    report = run_leakage_suite(
        cutoff=cutoff,
        frames={"demand": (train, "date")},  # solo lo observado <= corte
        split={"strategy": "expanding", "train_end": "2026-05-31", "val_start": "2026-06-01"},
        train=train,
        full=full,
        feature_fn=_feature_fn_honest,
        feature_cols=["mean_rn"],
    )
    assert report.clean


def test_inyeccion_rn_futuro_falla():
    df = _history()
    with pytest.raises(LeakageError) as exc:
        run_leakage_suite(cutoff=pd.Timestamp("2026-05-31"), frames={"demand": (df, "date")})
    assert "future_demand" in str(exc.value)


def test_inyeccion_pickup_futuro_falla():
    df = _history()
    with pytest.raises(LeakageError) as exc:
        run_leakage_suite(cutoff=pd.Timestamp("2026-04-30"), frames={"pickup": (df, "date")})
    assert "future_pickup" in str(exc.value)


def test_inyeccion_capacidad_futura_falla():
    df = _history()
    with pytest.raises(LeakageError) as exc:
        run_leakage_suite(cutoff=pd.Timestamp("2026-03-31"), frames={"capacity": (df, "date")})
    assert "future_capacity" in str(exc.value)


def test_inyeccion_feature_dataset_completo_falla():
    df = _history()
    cutoff = pd.Timestamp("2026-05-31")
    train = df.loc[df["date"] <= cutoff].copy()
    full = df.copy()

    leaky_stored = _feature_fn_leaky(full).loc[train.index]  # computado con full
    with pytest.raises(LeakageError) as exc:
        run_leakage_suite(
            cutoff=cutoff,
            frames={"demand": (train, "date")},
            train=leaky_stored,
            full=full,
            feature_fn=_feature_fn_leaky,  # la función que produjo las features
            feature_cols=["mean_rn"],
        )
    assert "whole_dataset" in str(exc.value)


def test_inyeccion_split_aleatorio_falla():
    with pytest.raises(LeakageError) as exc:
        run_leakage_suite(
            cutoff=pd.Timestamp("2026-05-31"),
            frames={"demand": (_history(), "date")},
            split={"strategy": "kfold", "n_splits": 5},
        )
    assert "random_split" in str(exc.value)


def test_rolling_con_futuro_detectado():
    df = _history(periods=40, end="2026-06-30")
    raw = pd.Series(np.arange(40, dtype=float))
    window = 5
    cutoff = pd.Timestamp("2026-06-30")

    # feature construida con ventana CENTRADA (leaky)
    df_leaky = df.assign(
        raw=raw,
        rn_feature=raw.rolling(window=window, min_periods=1, center=True).mean(),
    )
    finding = check_rolling_with_future(
        df_leaky, "date", "raw", "rn_feature", window, cutoff
    )
    assert finding is not None and finding.check == "rolling_includes_future"

    # feature construida con ventana retrospectiva (honesto) no señala
    df_honest = df.assign(
        raw=raw,
        rn_feature=raw.rolling(window=window, min_periods=1, center=False).mean(),
    )
    assert (
        check_rolling_with_future(df_honest, "date", "raw", "rn_feature", window, cutoff)
        is None
    )


def test_corrida_falla_no_publica_forecast(tmp_path):
    """3.4: leakage => la corrida falla y no escribe forecast."""
    from src.data.pipeline import forecast_output_path
    from src.validation.gate import enforce  # noqa: F401  (referencia de estilo)

    df = _history()
    run_dir = tmp_path / "run"
    with pytest.raises(LeakageError):
        run_leakage_suite(cutoff=pd.Timestamp("2026-05-31"), frames={"demand": (df, "date")})
        # si llegara aquí (no llega), escribiría el forecast:
        forecast_output_path(run_dir).write_text("should,not,happen\n")

    assert not forecast_output_path(run_dir).exists()

# --------------------- remediación: target en rolling features --------------


def test_rolling_con_target_de_la_propia_fila_detectado():
    """Una rolling que incluye el target de la fila es leakage de matriz ML."""
    from src.validation.leakage import (
        assert_no_target_in_rolling,
        check_rolling_target_inclusion,
        find_rolling_target_inclusions,
    )

    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    vals = pd.Series(np.arange(30, dtype=float) ** 1.3)
    contaminated = pd.DataFrame(
        {
            "property": ["corin"] * 30,
            "stay_date": dates,
            "room_nights": vals,
            "rolling_mean_7": vals.rolling(7, min_periods=1).mean(),  # incluye la fila
        }
    )
    finding = check_rolling_target_inclusion(
        contaminated, "room_nights", "rolling_mean_7", 7, "mean",
        date_col="stay_date", group_col="property",
    )
    assert finding is not None and finding.check == "rolling_includes_target"
    assert find_rolling_target_inclusions(contaminated, "room_nights", "stay_date", "property")
    with pytest.raises(LeakageError):
        assert_no_target_in_rolling(contaminated, "room_nights", "stay_date", "property")

    honest = contaminated.assign(
        rolling_mean_7=vals.shift(1).rolling(7, min_periods=1).mean()
    )
    assert (
        find_rolling_target_inclusions(honest, "room_nights", "stay_date", "property") == []
    )
    assert_no_target_in_rolling(honest, "room_nights", "stay_date", "property")


def test_features_de_demanda_pasan_check_de_target():
    """El motor post-remediación produce features sin el target de la fila."""
    from src.features.demand import add_demand_features
    from src.validation.leakage import assert_no_target_in_rolling

    rng = np.random.default_rng(11)
    n = 120
    df = pd.DataFrame(
        {
            "property": ["corin"] * n,
            "stay_date": pd.date_range("2026-01-01", periods=n),
            "room_nights": rng.integers(20, 90, size=n).astype(float),
        }
    )
    out, _ = add_demand_features(df)
    assert_no_target_in_rolling(out, "room_nights", "stay_date", "property")
