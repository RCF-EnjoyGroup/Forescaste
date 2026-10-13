"""Selección de campeón (spec ``evaluation``, tasks 6.7/6.8).

Criterio: WMAPE (principal) + MAE + estabilidad entre ventanas + penalización
de |Bias|. **El MAPE solo no decide**: un modelo con mejor MAPE pero bias
fuerte no gana (D4 del design).

Significancia (remediación): si se provee un panel de errores por fecha, la
ventaja del campeón sobre el subcampeón se contrasta con Diebold-Mariano; un
empate estadístico (p > 0.05) se desempata por parsimonia — entre dos modelos
estadísticamente indistinguibles opera el más simple (baseline primero).

Selección por propiedad con mínimo de ventanas; si no hay evidencia suficiente
→ fallback a la selección global ("sin selección propia").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation.metrics import diebold_mariano

BIAS_PENALTY = 0.5  # peso de |bias| relativo al WMAPE en el score compuesto
STABILITY_WEIGHT = 0.25  # peso de la volatilidad del WMAPE entre ventanas
MIN_FOLDS_PER_PROPERTY = 3
# composite ponderado por lead: los leads operativos (1-30) pesan ~2.8x por
# día que los de planeación (31-89) — un candidato plano no gana por
# "consistencia" en el rango largo mientras destruye el patrón del corto
LEAD_ALPHA = 0.65
LEAD_SPLIT = 30
# etiqueta de la fila agregada del portafolio en fold_metrics: NO participa
# de la selección global (efecto Simpson: el agregado oculta el error de las
# propiedades chicas) ni de la selección por propiedad
PORTFOLIO_LABEL = "PORTFOLIO"
# parsimonia para desempatar empates estadísticos (menor índice = más simple)
PARSIMONY_ORDER = [
    "seasonal_naive", "pickup_curve", "ets", "sarimax",
    "lightgbm", "xgboost", "catboost",
]


def parsimony_rank(model: str) -> int:
    """Índice de parsimonia: menor = más simple (primero en empate)."""
    name = str(model).lower()
    for i, m in enumerate(PARSIMONY_ORDER):
        if m == name:
            return i
    return len(PARSIMONY_ORDER)


@dataclass
class CandidateScore:
    model: str
    wmape: float
    mae: float
    bias: float
    mape: float
    stability: float  # std del WMAPE entre ventanas (menor = más estable)
    n_folds: int
    rmse: float = float("nan")
    composite: float = 0.0
    wmape_short: float = float("nan")  # leads 1-30 (operativo)
    wmape_long: float = float("nan")   # leads 31-89 (planeación)

    def __post_init__(self) -> None:
        # score compuesto: menor es mejor; el MAPE no entra (spec: no decide).
        # Con desagregación de lead: pondera operativo (0.65) vs planeación
        # (0.35) — un modelo plano en día-de-semana no gana por "consistencia"
        # en el rango largo si destruye el patrón operativo del corto.
        wmape_score = self.wmape
        if np.isfinite(self.wmape_short) and np.isfinite(self.wmape_long):
            wmape_score = LEAD_ALPHA * self.wmape_short + (1 - LEAD_ALPHA) * self.wmape_long
        self.composite = (
            wmape_score
            + BIAS_PENALTY * abs(self.bias)
            + STABILITY_WEIGHT * self.stability
        )


@dataclass
class SelectionResult:
    champion: str
    scores: list[CandidateScore]
    reason: str
    by_property: dict[str, str] = field(default_factory=dict)
    property_fallback: dict[str, str] = field(default_factory=dict)
    significance: dict | None = None

    @property
    def ranking(self) -> list[CandidateScore]:
        return sorted(self.scores, key=lambda s: s.composite)

    def table(self) -> pd.DataFrame:
        rows = [
            {
                "model": s.model,
                "wmape": s.wmape,
                "mae": s.mae,
                "rmse": s.rmse,
                "bias": s.bias,
                "mape": s.mape,
                "stability": s.stability,
                "n_folds": s.n_folds,
                "composite": s.composite,
                "wmape_short": s.wmape_short,
                "wmape_long": s.wmape_long,
            }
            for s in self.ranking
        ]
        return pd.DataFrame(rows)


def score_candidates(
    fold_metrics: pd.DataFrame,
    model_col: str = "model",
    fold_col: str = "fold",
) -> list[CandidateScore]:
    """Agrega métricas por ventana a un CandidateScore por modelo.

    ``fold_metrics``: columnas model, fold, mape, wmape, mae, bias.
    """
    required = {model_col, fold_col, "mape", "wmape", "mae", "bias"}
    missing = required - set(fold_metrics.columns)
    if missing:
        raise ValueError(f"columnas ausentes en fold_metrics: {sorted(missing)}")

    scores: list[CandidateScore] = []
    has_rmse = "rmse" in fold_metrics.columns
    has_lead_split = {"wmape_short", "wmape_long"} <= set(fold_metrics.columns)
    for model, grp in fold_metrics.groupby(model_col):
        stability = float(grp["wmape"].std()) if len(grp) > 1 else 0.0
        scores.append(
            CandidateScore(
                model=str(model),
                wmape=float(grp["wmape"].mean()),
                mae=float(grp["mae"].mean()),
                bias=float(grp["bias"].mean()),
                mape=float(grp["mape"].mean()),
                stability=stability,
                n_folds=int(len(grp)),
                rmse=float(grp["rmse"].mean()) if has_rmse else float("nan"),
                wmape_short=(
                    float(grp["wmape_short"].mean()) if has_lead_split else float("nan")
                ),
                wmape_long=(
                    float(grp["wmape_long"].mean()) if has_lead_split else float("nan")
                ),
            )
        )
    return scores


def _dm_between(
    error_panel: pd.DataFrame, model_a: str, model_b: str, horizon: int,
) -> tuple[str | None, dict | None]:
    """DM entre dos modelos sobre el panel de errores (pares alineados)."""
    if error_panel is None or error_panel.empty:
        return None, None
    cols = {"model", "stay_date", "actual", "forecast"}
    if not cols <= set(error_panel.columns):
        return None, None
    a = error_panel.loc[error_panel["model"] == model_a]
    b = error_panel.loc[error_panel["model"] == model_b]
    if a.empty or b.empty:
        return None, None
    keys = ["stay_date"]
    if "fold" in error_panel.columns:
        keys.append("fold")
    if "property" in error_panel.columns:
        keys.append("property")
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"))
    if len(merged) < 5 or not np.allclose(
        merged["actual_a"].to_numpy(dtype=float), merged["actual_b"].to_numpy(dtype=float)
    ):
        return None, None
    dm = diebold_mariano(
        merged["actual_a"].to_numpy(dtype=float),
        merged["forecast_a"].to_numpy(dtype=float),
        merged["forecast_b"].to_numpy(dtype=float),
        horizon=horizon,
    )
    better = None
    if dm.significant:
        better = model_b if dm.better == "b" else model_a
    return better, {
        "champion": model_a, "runner_up": model_b, **dm.to_dict(),
    }


def _without_portfolio_rows(
    fold_metrics: pd.DataFrame, property_col: str | None
) -> pd.DataFrame:
    """Excluye las filas agregadas de portafolio de la selección.

    El campeón global se decide con la evidencia por propiedad (media de
    WMAPEs por propiedad, cada una con el mismo peso): usar la fila
    ``PORTFOLIO`` comete el efecto Simpson — el WMAPE agregado refleja el
    error de las propiedades grandes y oculta el de las chicas.
    """
    if property_col is None or property_col not in fold_metrics.columns:
        return fold_metrics
    return fold_metrics.loc[fold_metrics[property_col] != PORTFOLIO_LABEL]


def select_champion(
    fold_metrics: pd.DataFrame,
    model_col: str = "model",
    fold_col: str = "fold",
    error_panel: pd.DataFrame | None = None,
    dm_horizon: int = 30,
    property_col: str | None = "property",
) -> SelectionResult:
    """6.7 — Elige el campeón global con evidencia out-of-sample.

    Con ``error_panel`` (errores por fecha al grano de selección), la ventaja
    del campeón sobre el subcampeón se contrasta con Diebold-Mariano: un
    empate estadístico se declara y se desempata por parsimonia. Con columna
    ``property``, la selección global usa solo las filas por propiedad
    (anti-Simpson).
    """
    fold_metrics = _without_portfolio_rows(fold_metrics, property_col)
    scores = score_candidates(fold_metrics, model_col, fold_col)
    if not scores:
        raise ValueError("sin candidatos para seleccionar")
    ranked = sorted(scores, key=lambda s: s.composite)
    champion = ranked[0]
    significance: dict | None = None

    if len(ranked) > 1 and error_panel is not None:
        runner_up = ranked[1]
        better, sig = _dm_between(error_panel, champion.model, runner_up.model, dm_horizon)
        significance = sig
        if sig is not None and not sig.get("significant"):
            # empate estadístico → desempate por parsimonia (más simple gana)
            if parsimony_rank(runner_up.model) < parsimony_rank(champion.model):
                champion = runner_up
                champion_note = (
                    f"; empate estadístico con '{ranked[0].model}' (DM p={sig.get('pvalue'):.3f}) "
                    f"→ desempate por parsimonia a favor de '{champion.model}'"
                )
            else:
                champion_note = (
                    f"; empate estadístico con '{runner_up.model}' (DM p={sig.get('pvalue'):.3f}) "
                    "→ campeón retenido por parsimonia"
                )
        elif better is not None:
            champion_note = f"; ventaja DM significativa sobre '{runner_up.model}' (p={sig.get('pvalue'):.4f})"
        else:
            champion_note = ""
    else:
        champion_note = ""

    # verificación de honestidad: el mejor MAPE no necesariamente gana
    best_mape = min(scores, key=lambda s: s.mape)
    reason = (
        f"campeón '{champion.model}' por score compuesto "
        f"(WMAPE={champion.wmape:.4f}, |Bias|={abs(champion.bias):.4f}, "
        f"estabilidad={champion.stability:.4f}){champion_note}; "
        f"mejor MAPE individual: '{best_mape.model}' "
        f"(el MAPE por sí solo no decide)"
    )
    return SelectionResult(
        champion=champion.model, scores=scores, reason=reason, significance=significance,
    )


def select_per_property(
    fold_metrics: pd.DataFrame,
    min_folds: int = MIN_FOLDS_PER_PROPERTY,
    model_col: str = "model",
    fold_col: str = "fold",
    property_col: str = "property",
    global_champion: str | None = None,
) -> SelectionResult:
    """6.8 — Selección por propiedad con mínimo de ventanas y fallback global.

    Propiedades sin ``min_folds`` ventanas por candidato caen al campeón
    global y se marcan en ``property_fallback`` como 'sin selección propia'.
    """
    fold_metrics = _without_portfolio_rows(fold_metrics, property_col)
    if global_champion is None:
        global_result = select_champion(fold_metrics, model_col, fold_col)
        global_champion = global_result.champion
        base_scores = global_result.scores
        reason = global_result.reason
    else:
        base_scores = score_candidates(fold_metrics, model_col, fold_col)
        reason = f"campeón global proporcionado: '{global_champion}'"

    by_property: dict[str, str] = {}
    fallback: dict[str, str] = {}

    for prop, grp in fold_metrics.groupby(property_col):
        n_folds_models = grp.groupby(model_col)[fold_col].nunique()
        # mínimo: todos los candidatos evaluados con >= min_folds ventanas
        eligible = n_folds_models[n_folds_models >= min_folds]
        usable = grp[grp[model_col].isin(eligible.index)]
        if usable.empty or usable[model_col].nunique() < 2:
            by_property[str(prop)] = global_champion
            fallback[str(prop)] = "sin selección propia (insuficiente evidencia)"
            continue
        scores = score_candidates(usable, model_col, fold_col)
        winner = min(scores, key=lambda s: s.composite)
        by_property[str(prop)] = winner.model

    return SelectionResult(
        champion=global_champion,
        scores=base_scores,
        reason=reason,
        by_property=by_property,
        property_fallback=fallback,
    )
