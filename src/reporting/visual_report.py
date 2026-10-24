"""Reporte visual HTML por corrida (interfaz de análisis/presentación).

Genera un archivo ``report.html`` autocontenido dentro de la carpeta de la
corrida, con los gráficos de revisión embebidos en base64 (no requiere
navegador, Jupyter ni conexión). Es una vista: toda la lógica vive en
``src/`` (design D1: los notebooks/reportes solo importan y presentan), y no
altera los 10 entregables del spec (reporting §35).

Secciones (todas defensivas ante datos ausentes):

- **EDA**: volumen diario del portafolio.
- **Pickup**: ocupación de los libros por lead time.
- **Benchmark**: WMAPE por modelo candidato con el campeón destacado.
- **Forecast**: pronóstico por propiedad con banda P10/P50/P90.
- **Forecast por propiedad**: small multiples con banda, techo de capacidad,
  reales cuando existen y el modelo que opera cada hotel.
- **Forecast vs reales**: superposición y dispersión cuando hay ``actual``.
- **Accuracy por propiedad**: barras de WMAPE + tabla con modelo y bias.
- **Revisiones T-90..T-1 por propiedad**: WMAPE del backtest muestreado y
  convergencia del error por banda de lead (una línea por propiedad).
- **Ocupación**: forecast contra el techo de capacidad.
"""

from __future__ import annotations

import base64
import io
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin backend interactivo: seguro en CI/servidor

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.breakdowns import lead_band as _lead_band


def _with_lead_band(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve ``df`` con una columna ``lead_band`` derivada de ``lead_days``.

    No muta la entrada. Si no hay ``lead_days`` numérico devuelve el frame
    original para que las secciones dependientes simplemente se omitan.
    """
    if "lead_band" in df.columns or "lead_days" not in df.columns:
        return df
    try:
        out = df.copy()
        out["lead_band"] = [_lead_band(v) for v in out["lead_days"]]
        return out
    except (TypeError, ValueError):
        return df

# paleta del sistema de diseño del reporte (papel frío + petróleo + semánticos)
# coherente entre HTML y matplotlib: los gráficos y la página hablan el mismo idioma
_COLORS = {
    "ink":      "#0B2239",  # tinta océano (títulos, texto fuerte)
    "primary":  "#0E7490",  # petróleo (líneas de forecast, acento del sistema)
    "forecast": "#0E7490",  # alias semántico: serie pronosticada
    "actual":   "#C2410C",  # terracota profundo (reales — cálido contra el frío)
    "band":     "#A8D8E8",  # banda P10-P90 (petróleo claro)
    "capacity": "#94A3B8",  # techo de capacidad (slate)
    "champ":    "#15803D",  # verde campeón
    "other":    "#D5DEE7",  # barras no-campeón (slate claro)
    "accent2":  "#B45309",  # ámbar (categoría secundaria)
    "accent3":  "#6D28D9",  # ciruela (categoría terciaria)
    "grid":     "#E2E8F0",
    "text":     "#334155",
    "muted":    "#64748B",
}


def _apply_chart_style() -> None:
    """Sistema visual de los gráficos — coherente con el diseño de la página.

    Reglas de la casa (se aplican a TODAS las gráficas del reporte): fondo
    blanco, spines superior/derecho fuera, sutil grilla horizontal, títulos
    alineados a la izquierda en tinta océano y tipografía 9pt — el lector
    debe leer el gráfico, no el adorno.
    """
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": _COLORS["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": _COLORS["text"],
        "axes.labelcolor": "#475569",
        "axes.labelsize": 9,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": 600,
        "axes.titlecolor": _COLORS["ink"],
        "axes.titlelocation": "left",
        "figure.dpi": 110,
    })


def _fig_to_base64(fig: plt.Figure) -> str:
    """Figura -> PNG base64 para embeber en ``<img>`` sin archivos sueltos."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------
# guías de lectura por sección (texto de presentación, no lógica)
# ------------------------------------------------------------------

_EXPLAINERS: dict[str, str] = {
    "Forecast": (
        "<b>Qué ves:</b> el pronóstico diario total del portafolio (P50 = mediana) "
        "con la banda de incertidumbre P10–P90 sombreada. <b>Cómo leerla:</b> hay "
        "80% de probabilidad de que el real caiga dentro del área sombreada (10% "
        "por debajo de P10, 10% por encima de P90). Banda angosta = confianza "
        "(típico a leads cortos, cuando ya hay reservas en libros); banda ancha = "
        "incertidumbre real. El zigzag semanal (picos fin de semana) debe "
        "aparecer: una línea plana sería mala señal."
    ),
    "Forecast vs Real": (
        "<b>Qué ves:</b> pronóstico y reales superpuestos (solo en corridas de "
        "verificación, donde la ventana ya tiene reales). <b>Cómo leerla:</b> "
        "cuanto más pegadas caminen las líneas, mejor. Un tramo donde una queda "
        "sistemáticamente arriba/abajo no es ruido: es sesgo visible a simple "
        "vista (se cuantifica en el KPI Bias). Reales saliéndose de la banda con "
        "frecuencia = banda mal calibrada (ver panel de validación)."
    ),
    "Forecast por propiedad": (
        "<b>Qué ves:</b> una mini-gráfica por hotel con el P50 (línea), la banda "
        "P10–P90 (sombra), el techo de capacidad y el modelo que opera ese hotel "
        "en el título. <b>Por qué importa:</b> el total del portafolio esconde "
        "todo — aquí se ve que cada hotel tiene su propia escala de error (la "
        "banda se calibra por propiedad, no con el promedio). <b>Qué mirar:</b> "
        "si el P50 toca el techo de capacidad en fechas específicas, el hotel se "
        "pronostica lleno (señal de pricing)."
    ),
    "Benchmark": (
        "<b>Qué ves:</b> el bake-off de los 7 candidatos: WMAPE por modelo en 4 "
        "ventanas trimestrales del año previo al corte, evaluadas sin ver el "
        "futuro (sin fugas). Menor = mejor; el campeón va destacado. <b>Cómo "
        "leerla:</b> si el subcampeón está cerca, la prueba Diebold-Mariano "
        "decide: un empate estadístico se desempata por parsimonia (el más "
        "simple opera); una ventaja significativa cambia el campeón. El veredicto "
        "exacto está en el panel de validación."
    ),
    "Pickup": (
        "<b>Qué ves:</b> room nights ya reservados según los días que faltan "
        "para la estancia. <b>Cómo leerla:</b> la curva típica es cóncava — se "
        "acumula rápido al cierre. Es el corazón del modelo campeón: a lo ya "
        "reservado le suma lo que históricamente se recoge a esa anticipación. "
        "A leads largos (30+) hay poco reservado: por eso el pronóstico a 60–90 "
        "días depende de la señal estacional y carga con más incertidumbre."
    ),
    "Ocupación": (
        "<b>Qué ves:</b> el pronóstico traducido a porcentaje del inventario "
        "disponible. <b>Guía rápida:</b> sostenido > 80% = temporada alta "
        "(presión de precios); < 40% = temporada baja (descuentos y marketing "
        "importan). Es la vista en el lenguaje del negocio."
    ),
    "EDA": (
        "<b>Qué ves:</b> el histórico real de demanda — el mundo sobre el que "
        "todo lo demás se entrena y evalúa. <b>Qué mirar:</b> estacionalidad "
        "anual, el efecto COVID (2020–21) y el crecimiento del portafolio "
        "(más hoteles abriendo = saltos de nivel). Por eso el drift se compara "
        "año contra año, no contra el histórico completo."
    ),
    "Dispersión": (
        "<b>Qué ves:</b> cada punto es un día-hotel: eje X = real, eje Y = "
        "pronosticado. <b>Cómo leerla:</b> puntos sobre la diagonal = "
        "pronóstico perfecto; nube pegada a la diagonal = buen modelo. Si la "
        "nube queda sistemáticamente debajo de la diagonal en la derecha (días "
        "grandes), el modelo sub-estima la demanda alta — el patrón del Bias "
        "negativo."
    ),
    "Accuracy por propiedad": (
        "<b>Qué ves:</b> barras de WMAPE por hotel (verde < 15%, azul < 30%) y "
        "la tabla exacta con el modelo que opera, el WMAPE semanal y el bias. "
        "<b>Por qué el semanal:</b> en hoteles pequeños el error diario es ruido "
        "de denominador (±3 RN con reales de 5 ya son 60%); agregado semanal, "
        "el patrón se ve limpio — <b>para decisiones de negocio, el semanal es "
        "la vista honesta</b>. La columna de cobertura dice si la banda de ese "
        "hotel contiene el 80% nominal de los reales."
    ),
    "Revisiones T-90..T-1 · por propiedad": (
        "<b>Qué ves:</b> el backtest operativo: fechas de estancia pasadas "
        "pronosticadas como si faltaran 90, 60, 30, 14, 7, 3 y 1 días, con el "
        "campeón operando cada revisión. <b>Cómo leer la convergencia:</b> es "
        "la física del problema — a 90 días hay poca información (error alto), "
        "al cierre casi todo está reservado (error mínimo). Una línea que no "
        "converge indica un problema en ese hotel. <b>La pregunta de negocio "
        "que responde:</b> ¿con cuánta anticipación puedo confiar en el "
        "pronóstico? Usa el WMAPE de la banda de lead de tu horizonte de "
        "decisión (p. ej. 14-29 para pricing con 2 semanas de margen), no el "
        "global."
    ),
}


def _explainer(name: str) -> str:
    """Bloque plegable con la guía de lectura de la sección (si existe)."""
    text = _EXPLAINERS.get(name)
    if not text:
        return ""
    return (
        f'<details class="expl"><summary>¿Cómo leer esta sección?</summary>'
        f'<p>{text}</p></details>'
    )


# ------------------------------------------------------------------
# glosario + panel de interpretación y validación
# ------------------------------------------------------------------


def _glossary_card() -> str:
    terms = [
        ("WMAPE", "Error ponderado por volumen: de cada 100 RN reales, cuántos "
                  "se equivocó el pronóstico. La métrica principal (pesa más "
                  "los días y hoteles grandes)."),
        ("MAPE", "Error porcentual por día promediado simple (cada día vale "
                 "igual). Siempre mayor que el WMAPE; no compara contra el "
                 "de otras ventanas."),
        ("Bias", "Sesgo sistemático: negativo = tiende a sub-estimar; "
                 "positivo = sobre-estimar. Lo ideal, cerca de 0."),
        ("Lead days", "Días que faltan para la estancia. T-90..T-1 = el mismo "
                      "día pronosticado cuando faltaban 90, 60, 30… y 1 días."),
        ("P10/P50/P90", "Percentiles del pronóstico: la banda contiene el real "
                        "el 80% de las veces si está bien calibrada. P50 es el "
                        "valor central publicado."),
        ("Fold", "Ventana de evaluación del benchmark (trimestral, del año "
                 "previo al corte, sin solapar la ventana reportada)."),
        ("Vintage / forecast_date", "La fecha «as-of» con la que se PUBLICÓ un "
                                    "forecast (el corte de esa corrida). El "
                                    "tracking vintage puntuía el forecast tal "
                                    "como se publicó ese día, sin ventaja "
                                    "retrospectiva: por eso en el detalle "
                                    "`vintage_tracking_detail.csv` la columna no "
                                    "cambia entre filas — todo el archivo evalúa "
                                    "UNA publicación; cambia en el siguiente "
                                    "ciclo cuando se puntúa la corrida más "
                                    "recente."),
    ]
    rows = "".join(
        f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in terms
    )
    return (
        '<section class="card wide glossary"><h2>Glosario rápido</h2>'
        '<table class="tbl"><thead><tr><th style="width:110px">Término</th>'
        "<th>Significado</th></tr></thead><tbody>" + rows + "</tbody></table>"
        "</section>"
    )


def _chip(level: str, text: str) -> str:
    """Lámpara de veredicto: ok (verde) / warn (ámbar) / bad (rojo) / info (gris).

    El punto de color codifica la severidad; el texto queda plano y
    profesional (sin emojis — este reporte se imprime y se presenta).
    """
    return f'<span class="chip {level}">{text}</span>'


def _vintage_grupo(vintage_tracking: dict | None, bandas: tuple[str, ...]) -> str:
    """Cita la medición MÁS FRESCA del grupo cuando la añada la cubre.

    El veredicto del grupo usa la evaluación de la corrida (en modo futuro,
    el backtest as-of); la añada publicada reciente es la medición real
    más reciente de esos leads — el lector merece ver ambas cuando
    difieren (p. ej. corto eval 10% vs añada real 5.5%).
    """
    rows = (vintage_tracking or {}).get("by_lead_band") or []
    by_band = {str(r.get("lead_band")): r for r in rows if r.get("wmape") is not None}
    grupales = [by_band[b] for b in bandas if b in by_band]
    if not grupales:
        return ""
    abs_err = sum(float(r["mae"]) * int(r.get("n_obs", 0)) for r in grupales)
    denom = sum(
        float(r["mae"]) * int(r.get("n_obs", 0)) / max(float(r["wmape"]), 1e-9)
        for r in grupales
    )
    n = sum(int(r.get("n_obs", 0)) for r in grupales)
    if denom <= 0 or n < 10:
        return ""
    return f" · añada publicada reciente: {abs_err / denom:.1%} (n={n}, real)"


def _chart_track_record(track: list[dict] | None) -> str | None:
    """WMAPE por añada — el expediente en una vista."""
    if not track:
        return None
    labels = [
        f"{e.get('cutoff', '—')[:10]}\n(n={e.get('n_obs', 0)})"
        for e in track
    ]
    wmape = [e.get("wmape") for e in track]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.bar(labels, wmape, color=[_COLORS["primary"]] * len(track))
    for bar, w in zip(bars, wmape):
        if np.isfinite(w):
            ax.text(bar.get_x() + bar.get_width() / 2, w, f"{w:.1%}",
                    ha="center", va="bottom", fontsize=9, color=_COLORS["ink"])
    ax.set_title("Expediente · WMAPE del forecast publicado por añada (corte de publicación)",
                 loc="left", fontsize=10.5)
    ax.set_ylabel("WMAPE")
    ax.grid(alpha=0.25, axis="y")
    return _fig_to_base64(fig)


def _table_track_record(track: list[dict] | None) -> str:
    """Tabla del expediente: una fila por añada graduada."""
    if not track:
        return '<p class="muted">Sin añadas graduadas aún — se activa con el primer vintage.</p>'
    rows = []
    for e in track:
        cov = e.get("band_coverage")
        cov_cell = f"{cov:.0%}" if cov is not None and np.isfinite(cov) else "—"
        ts = e.get("tracking_signal_worst") or {}
        ts_cell = (
            f"{ts['value']:+.1f} ({ts['property']})"
            if ts and np.isfinite(ts.get("value", float("nan")))
            else "—"
        )
        rows.append(
            f"<tr><td>{e.get('cutoff', '—')}</td>"
            f"<td>{e['window'].get('start', '—')} → {e['window'].get('end', '—')}</td>"
            f"<td>{e.get('n_obs', 0)}</td>"
            f"<td>{e.get('wmape', float('nan')):.1%}</td>"
            f"<td>{e.get('bias', float('nan')):+.1%}</td>"
            f"<td>{cov_cell}</td>"
            f"<td>{ts_cell}</td></tr>"
        )
    return (
        '<table class="tbl"><thead><tr><th>Publicada (corte)</th>'
        "<th>Ventana graduada</th><th>n obs</th><th>WMAPE</th>"
        "<th>Bias</th><th>Cobertura banda</th><th>TS peor</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _explainer_track_record() -> str:
    return (
        '<details class="expl"><summary>¿Cómo leer esta sección?</summary>'
        "<p><b>Qué ves:</b> el expediente del forecast publicado — cada "
        "añada (publicación) graduada contra los reales que se materializaron, "
        "con la graduación más reciente de cada una (la que más reales "
        "acumuló). <b>Cómo leerla:</b> es la respuesta a la pregunta de "
        "dueños «¿este forecast es confiable y está mejorando?» — WMAPE por "
        "añada con su n (añadas recientes tienen menos días materializados: "
        "leer con el n a la vista). <b>Fuente:</b> el registro entre "
        "corridas; crece solo con cada refresh de datos.</p></details>"
    )


def _validation_panel(
    *,
    metrics: dict | None,
    benchmark: pd.DataFrame | None,
    champion: str | None,
    significance: dict | None,
    interval_coverage: dict | None,
    backtest_report: dict | None,
    cold_start: dict[str, str] | None,
    drift_alerts: list | None,
    monitoring_state: dict | None,
    bias_correction: dict | None,
    vintage_tracking: dict | None = None,
    forecast_revision: dict | None = None,
    lead_band_breakdown: list[dict] | None = None,
    benchmark_bands: dict | None = None,
    tracking_signal_alert: float | None = None,
    track_record: list[dict] | None = None,
) -> str:
    """Tabla de veredictos: métrica | valor | criterio | lectura.

    Umbrales INTERPRETATIVOS de presentación (ayudan a leer el reporte); los
    gates duros de la corrida viven en los criterios de aceptación del
    runner y en el gate de calidad de datos.
    """
    rows: list[str] = []

    def _row(name: str, value: str, criterio: str, chip: str) -> None:
        rows.append(
            f"<tr><td>{name}</td><td><b>{value}</b></td>"
            f"<td>{criterio}</td><td>{chip}</td></tr>"
        )

    # --- WMAPE de la ventana (con contexto del benchmark anual)
    if metrics and "wmape" in metrics:
        w = float(metrics["wmape"])
        nivel = _chip("ok", "Bien") if w < 0.15 else (
            _chip("warn", "Revisar") if w < 0.25 else _chip("bad", "Deteriorado")
        )
        folds_note = ""
        if benchmark is not None and not benchmark.empty and champion:
            champ = benchmark.loc[benchmark["model"] == champion]
            if not champ.empty and "wmape" in champ.columns:
                folds_w = float(champ["wmape"].mean())
                folds_note = (
                    f" · expectativa anual del campeón (folds): {folds_w:.1%}"
                )
        _row(
            "WMAPE (ventana)", f"{w:.1%}",
            "mezcla horizontes (el desglose contra banda industria va por "
            "grupo, abajo)" + folds_note,
            nivel,
        )

    # --- Precisión por horizonte contra banda de la INDUSTRIA (resort)
    # El pooling entre bandas del grupo es honesto: Σ|e| = Σ(mae·n) y
    # Σ|actual| = Σ(mae·n)/wmape → WMAPE del grupo ponderado por volumen.
    if lead_band_breakdown and benchmark_bands:
        grupos: list[tuple[str, tuple[str, ...], str]] = [
            ("corto (≤6d)", ("1-2", "3-6"), "short"),
            ("medio (7-29d)", ("7-13", "14-29"), "mid"),
            ("largo (30d+)", ("30-59", "60-89", "90+"), "long"),
        ]
        by_band = {
            str(r.get("lead_band")): r for r in lead_band_breakdown
            if r.get("wmape") is not None
        }
        for label, bandas, key in grupos:
            rows_grp = [by_band[b] for b in bandas if b in by_band]
            if not rows_grp:
                continue
            abs_err = sum(
                float(r["mae"]) * int(r.get("n_obs", 0)) for r in rows_grp
            )
            denom = sum(
                float(r["mae"]) * int(r.get("n_obs", 0)) / max(float(r["wmape"]), 1e-9)
                for r in rows_grp
            )
            n_grp = sum(int(r.get("n_obs", 0)) for r in rows_grp)
            if denom <= 0 or n_grp < 10:
                continue
            w_grp = abs_err / denom
            banda = benchmark_bands.get(key) or [0.25, 0.30]
            ok_hasta, deteriorado_desde = float(banda[0]), float(banda[1])
            nivel = (
                _chip("ok", "En banda") if w_grp < ok_hasta
                else (_chip("warn", "Ajustar") if w_grp < deteriorado_desde
                      else _chip("bad", "Deteriorado"))
            )
            _row(
                f"WMAPE {label} vs banda industria",
                f"{w_grp:.1%} (n={n_grp})",
                f"banda resort: < {ok_hasta:.0%} en banda · "
                f"{ok_hasta:.0%}-{deteriorado_desde:.0%} ajustar · "
                f"> {deteriorado_desde:.0%} deteriorado (estándar operativo "
                "HospitalityOS 2026, perfil estacional)"
                + _vintage_grupo(vintage_tracking, bandas),
                nivel,
            )

    # --- Bias
    if metrics and "bias" in metrics:
        b = float(metrics["bias"])
        nivel = _chip("ok", "Calibrado") if abs(b) < 0.05 else (
            _chip("warn", "Sesgo visible") if abs(b) < 0.10
            else _chip("bad", "Sesgo fuerte")
        )
        _row(
            "Bias", f"{b:+.1%}",
            "|sesgo| < 5% calibrado · < 10% visible · > 10% fuerte "
            "(negativo = sub-estima)",
            nivel,
        )

    # --- Cobertura de banda
    if interval_coverage and (interval_coverage.get("global") or {}).get("n_obs"):
        g = interval_coverage["global"]
        cov = float(g["coverage"])
        nominal = float(g.get("nominal", 0.80))
        desv = abs(cov - nominal)
        nivel = _chip("ok", "Calibrada") if desv <= 0.05 else (
            _chip("warn", "Ajustar") if desv <= 0.10 else _chip("bad", "Descalibrada")
        )
        peor = ""
        by_prop = interval_coverage.get("by_property") or []
        fuera = [
            r for r in by_prop
            if int(r.get("n_obs", 0)) >= 30
            and abs(float(r.get("coverage", nominal)) - nominal) > 0.05
        ]
        if fuera:
            peor = "; ".join(
                f"{r['property']} {float(r['coverage']):.0%}"
                for r in sorted(fuera, key=lambda r: abs(r["coverage"] - nominal),
                                reverse=True)
            )
            peor = f" · fuera de ±5pp: {peor}"
        _row(
            "Cobertura banda P10–P90", f"{cov:.1%}",
            f"debe contener el {nominal:.0%} nominal de los reales (±5pp "
            f"calibrada, ±10pp ajustar){peor}",
            nivel,
        )
    elif interval_coverage and interval_coverage.get("note"):
        _row(
            "Cobertura banda P10–P90", "—",
            "no medible en corrida futura (la banda vive sobre el horizonte, "
            "que aún no tiene reales); la cobertura por vintage se mide al "
            "materializarse los reales",
            _chip("info", "N/D"),
        )

    # --- Convergencia del backtest (lead 1-2 vs 60-89)
    if backtest_report:
        bands = {r["lead_band"]: r for r in backtest_report.get("by_lead_band") or []}
        short = bands.get("1-2")
        long_ = bands.get("60-89")
        if short and long_:
            ratio = float(long_["wmape"]) / max(float(short["wmape"]), 1e-9)
            nivel = _chip("ok", "Converge") if ratio >= 1.5 else (
                _chip("warn", "Convergencia débil") if ratio >= 1.1
                else _chip("bad", "No converge")
            )
            _row(
                "Convergencia del error",
                f"{float(short['wmape']):.1%} (lead 1-2) vs "
                f"{float(long_['wmape']):.1%} (60-89)",
                "el error al cierre debe ser mucho menor que a 90 días "
                f"(razón {ratio:.1f}×)",
                nivel,
            )

    # --- Significancia del campeón
    if significance:
        pv = significance.get("pvalue")
        rival = significance.get("runner_up", "?")
        if pv is not None:
            if significance.get("significant"):
                nivel = _chip("ok", "Ventaja significativa")
                crit = f"DM p={float(pv):.3f} < 0.05 vs '{rival}'"
            else:
                nivel = _chip("info", "Empate → parsimonia")
                crit = (
                    f"DM p={float(pv):.3f} ≥ 0.05 vs '{rival}': estadísticamente "
                    "indistinguibles, opera el más simple"
                )
            _row("Significancia del campeón", champion or "—", crit, nivel)

    # --- Cold start
    if cold_start:
        props = ", ".join(f"{p} ({v.split(':')[0]})" for p, v in sorted(cold_start.items()))
        _row(
            "Cold start (provisional)", props,
            "propiedades con < 365d de histórico: su forecast es provisional "
            "y no debe usarse para planeación sin esa salvedad",
            _chip("warn", "Provisional"),
        )

    # --- Drift YoY
    if drift_alerts is not None:
        demanda = [a for a in drift_alerts if str(a.get("metric", "")).startswith("demand_psi")]
        nivel = _chip("ok", "Sin drift") if not demanda else _chip(
            "warn", f"{len(demanda)} hallazgo(s) YoY"
        )
        detalle = (
            "; ".join(str(a.get("message", "")) for a in demanda[:3])
            if demanda
            else "ninguna propiedad cambió de nivel vs el año pasado"
        )
        _row(
            "Drift de demanda (YoY)", f"{len(demanda)} alerta(s)",
            "por propiedad contra la misma ventana un año atrás (PSI) — sin "
            "fatiga de alertas. " + detalle,
            nivel,
        )

    # --- Vintage: el forecast publicado contra los reales que llegaron
    if vintage_tracking and vintage_tracking.get("global"):
        vt = vintage_tracking
        w = float(vt["global"].get("wmape", float("nan")))
        n = int(vt.get("n_obs", 0))
        window = vt.get("window") or {}
        nivel = (
            _chip("ok", "Bien") if w < 0.15
            else (_chip("warn", "Revisar") if w < 0.25 else _chip("bad", "Deteriorado"))
        )
        band = (vt.get("interval_coverage") or {}).get("global") or {}
        extra = (
            f" · la banda publicada cubrió {float(band.get('coverage', float('nan'))):.0%} "
            f"(nominal {float(band.get('nominal', 0.8)):.0%})"
            if band else ""
        )
        detalle = (
            " — detalle por estancia en vintage_tracking_detail.csv"
            if vt.get("detail_csv") else ""
        )
        annada = (
            f" · añada: {vt['published_run_id']}" if vt.get("published_run_id") else ""
        )
        _row(
            "Forecast publicado vs reales (vintage)", f"{w:.1%} (n={n})",
            f"el forecast publicado por la corrida anterior, puntuado contra los "
            f"reales ya materializados ({window.get('start', '—')} → "
            f"{window.get('end', '—')}); WMAPE < 15% bien · 15–25% revisar{extra}"
            f"{annada}{detalle}",
            nivel,
        )

    # --- Tracking signal del forecast publicado (control de sesgo acumulado)
    # POR PROPIEDAD (control de cada hotel sobre la añada, como el scorecard
    # de la industria por departamento): la TS global escala con n y no
    # discrimina; la de cada propiedad sí.
    ts_threshold = float(tracking_signal_alert or 4.0)
    ts_rows = (vintage_tracking or {}).get("by_property") or []
    ts_rows = [r for r in ts_rows if r.get("tracking_signal") is not None]
    if ts_rows:
        provisional = {str(p) for p in (cold_start or {})}
        peor = ts_rows[0]
        fuera = [r for r in ts_rows if abs(float(r["tracking_signal"])) > 2 * ts_threshold]
        sesgo = [
            r for r in ts_rows
            if ts_threshold < abs(float(r["tracking_signal"])) <= 2 * ts_threshold
        ]
        detalle = " · ".join(
            f"{r['property']}{'⚠' if r['property'] in provisional else ''} "
            f"{float(r['tracking_signal']):+.1f}"
            for r in ts_rows
        )
        if fuera:
            nivel = _chip("bad", "Fuera de control")
        elif sesgo:
            nivel = _chip("warn", "Sesgo acumulado")
        else:
            nivel = _chip("ok", "En control")
        nota_prov = (
            " · ⚠ = cold start provisional (sesgo documentado hasta acumular histórico)"
            if provisional & {r["property"] for r in ts_rows}
            else ""
        )
        _row(
            "Tracking signal por propiedad (añada)",
            f"{float(peor['tracking_signal']):+.1f} ({peor['property']}, la peor)",
            f"|TS| ≤ {ts_threshold:.0f} en control — error con signo acumulado ÷ "
            "desviación media por hotel; crece cuando una propiedad yerra "
            f"consistente en UNA dirección (el patrón que el MAPE esconde). "
            f"{detalle}{nota_prov}",
            nivel,
        )

    # --- Revisión entre corridas
    if forecast_revision:
        d = float(forecast_revision.get("delta_pct", 0.0))
        nivel = (
            _chip("ok", "Estable") if abs(d) <= 0.05
            else (_chip("warn", "Movimiento notable") if abs(d) <= 0.15
                  else _chip("bad", "Cambio fuerte"))
        )
        _row(
            "Revisión entre corridas", f"{d:+.1%}",
            f"delta del forecast actual vs el publicado por la corrida anterior "
            f"sobre {int(forecast_revision.get('n_stays', 0))} estancias comunes "
            "aún futuras; |Δ| ≤ 5% estable · ≤ 15% notable",
            nivel,
        )

    # --- Expediente del forecast publicado (tendencia entre añadas)
    if track_record:
        n_annadas = len(track_record)
        detalle_exp = " · ".join(
            f"añada {e.get('cutoff', '—')[:10]}: {e.get('wmape', float('nan')):.1%} (n={e.get('n_obs', 0)})"
            for e in track_record
        )
        if n_annadas < 2:
            nivel = _chip("info", "Acumulando")
            tendencia = "una sola añada graduada — la tendencia llega con la segunda"
        else:
            ultima = float(track_record[-1]["wmape"])
            previas = [float(e["wmape"]) for e in track_record[:-1]
                       if np.isfinite(float(e.get("wmape", float("nan"))))]
            media_previa = sum(previas) / len(previas) if previas else float("nan")
            if np.isfinite(media_previa):
                delta_rel = (ultima - media_previa) / media_previa
                if delta_rel <= -0.15:
                    nivel = _chip("ok", "Mejorando")
                    tendencia = f"última añada {delta_rel:+.0%} vs el promedio de las previas"
                elif delta_rel < 0.15:
                    nivel = _chip("ok", "Estable")
                    tendencia = f"última añada {delta_rel:+.0%} vs el promedio de las previas (dentro de ±15%)"
                else:
                    nivel = _chip("warn", "Revisar")
                    tendencia = f"última añada {delta_rel:+.0%} vs el promedio de las previas"
            else:
                nivel = _chip("info", "Acumulando")
                tendencia = "sin comparación válida aún"
            # salvedad de n: añadas con pocos días materializados son ruido
            n_ultima = int(track_record[-1].get("n_obs", 0))
            if n_ultima < 30:
                tendencia += f" — OJO: la última añada solo tiene n={n_ultima}, leer con cautela"
        _row(
            "Expediente del forecast publicado", f"{n_annadas} añada{'s' if n_annadas != 1 else ''} graduada{'s' if n_annadas != 1 else ''}",
            f"{tendencia}. {detalle_exp}",
            nivel,
        )

    # --- Estado entre corridas (registro)
    if monitoring_state is not None:
        prev = monitoring_state.get("previous_run_id")
        nuevos = monitoring_state.get("new_records")
        if prev:
            crit = (
                f"corrida anterior: {prev} · {monitoring_state.get('days_since_last')}d "
                f"desde el último entrenamiento · {nuevos} registros nuevos"
            )
            nivel = _chip("ok", "Memoria activa")
        else:
            crit = "primera corrida con registro: la próxima compara contra esta"
            nivel = _chip("info", "Sin historial")
        _row("Registro entre corridas", "—", crit, nivel)

    # --- Corrección de bias
    if bias_correction and bias_correction.get("n_cells"):
        n_corr = int(bias_correction.get("n_corrected", 0))
        n_tot = int(bias_correction.get("n_cells", 0))
        _row(
            "Corrección de bias (folds)", f"{n_corr}/{n_tot} celdas",
            "aprendida de los residuos del forecast operativo por "
            "(propiedad × banda), con gate de consistencia entre folds y "
            "shrinkage; celdas inestables no se tocan",
            _chip("info", "Evidencia de folds"),
        )

    if not rows:
        return ""
    return (
        '<section class="card wide validation"><h2>Interpretación y validación</h2>'
        '<table class="tbl verdict"><thead><tr><th>Métrica</th><th>Valor</th>'
        "<th>Criterio / contexto</th><th>Veredicto</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
        '<p style="font-size:0.78em;color:#64748b;margin:8px 0 0 0">Veredictos '
        "interpretativos de lectura; los gates duros de la corrida (fugas, "
        "restricción de capacidad, calidad de datos, coherencia de "
        "entregables) viven en los criterios de aceptación del manifiesto.</p>"
        "</section>"
    )


# ------------------------------------------------------------------
# gráficos
# ------------------------------------------------------------------


def _chart_eda(demand: pd.DataFrame | None) -> str | None:
    """Volumen diario agregado del portafolio."""
    if demand is None or demand.empty:
        return None
    d = demand.copy()
    if not {"stay_date", "room_nights"}.issubset(d.columns):
        return None
    d["stay_date"] = pd.to_datetime(d["stay_date"], errors="coerce")
    daily = d.groupby("stay_date")["room_nights"].sum().sort_index()
    if daily.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(daily.index, daily.values, color=_COLORS["primary"], lw=1.2)
    ax.set_title("EDA · Room nights diarios del portafolio", loc="left", fontsize=11)
    ax.set_ylabel("RN")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _chart_pickup(fc: pd.DataFrame) -> str | None:
    """Ocupación de los libros (current booked) por bandas de lead time."""
    if not {"lead_band", "current_booked_rn"}.issubset(fc.columns):
        return None
    work = fc.copy()
    work["current_booked_rn"] = pd.to_numeric(work["current_booked_rn"], errors="coerce")
    work = work.dropna(subset=["current_booked_rn"])
    if work.empty:
        return None
    order = ["1-2", "3-6", "7-13", "14-29", "30-59", "60-89", "90+"]
    present = [b for b in order if b in set(work["lead_band"])]
    if not present:
        return None
    grouped = work.groupby("lead_band")["current_booked_rn"].mean().reindex(present)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(grouped.index, grouped.values, color=_COLORS["forecast"], alpha=0.85)
    ax.set_title("Pickup · RN ya reservados (promedio) por anticipación", loc="left", fontsize=11)
    ax.set_ylabel("RN reservados")
    ax.grid(alpha=0.25, axis="y")
    return _fig_to_base64(fig)


def _resolve_champion(
    benchmark: pd.DataFrame, champion: str | None
) -> str | None:
    """Campeón a destacar: columna ``is_champion`` de la tabla, o el explícito.

    Doble fuente (columna primero, parámetro como fallback): la vista no
    debe perder el realce del campeón cuando la tabla viene reconstruida
    (p. ej. ``scripts/regen_report.py`` desde model_benchmark.md).
    """
    if "is_champion" in benchmark.columns:
        mask = benchmark["is_champion"].astype(bool)
        if mask.any():
            return str(benchmark.loc[mask, "model"].iloc[0])
    return champion


def _chart_benchmark(
    benchmark: pd.DataFrame | None, champion: str | None = None
) -> str | None:
    """WMAPE por modelo candidato, destacando al campeón."""
    if benchmark is None or benchmark.empty or "wmape" not in benchmark.columns:
        return None
    b = benchmark.copy()
    b = b.dropna(subset=["wmape"])
    if b.empty:
        return None
    champ = _resolve_champion(b, champion)
    colors = [_COLORS["champ"] if m == champ else _COLORS["other"] for m in b["model"]]

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(b["model"], b["wmape"], color=colors)
    ax.set_title("Benchmark · WMAPE por modelo (menor = mejor)", loc="left", fontsize=11)
    ax.set_ylabel("WMAPE")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.25, axis="y")
    if champ:
        ax.text(0.99, 0.95, f"campeón: {champ}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=_COLORS["champ"])
    return _fig_to_base64(fig)


def _chart_forecast(fc: pd.DataFrame) -> str | None:
    """Serie de forecast agregada por fecha con banda P10/P50/P90."""
    if "stay_date" not in fc.columns:
        return None
    work = fc.copy()
    work["stay_date"] = pd.to_datetime(work["stay_date"], errors="coerce")
    work["final_forecast_rn"] = pd.to_numeric(work.get("final_forecast_rn"), errors="coerce")
    if work["final_forecast_rn"].notna().sum() == 0:
        return None
    daily = work.groupby("stay_date").sum(numeric_only=True).sort_index()

    fig, ax = plt.subplots(figsize=(9, 3.4))
    p10 = daily.get("p10_forecast")
    p90 = daily.get("p90_forecast")
    if p10 is not None and p90 is not None and p10.notna().any():
        ax.fill_between(daily.index, p10, p90, color=_COLORS["band"], alpha=0.5, label="P10–P90")
    ax.plot(daily.index, daily["final_forecast_rn"], color=_COLORS["forecast"], lw=1.8, label="Forecast")
    ax.set_title("Forecast · Room nights del portafolio con banda de incertidumbre", loc="left", fontsize=11)
    ax.set_ylabel("RN")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _chart_forecast_vs_actual(fc: pd.DataFrame, actual: pd.Series | None) -> str | None:
    """Superposición forecast vs reales (solo cuando hay reales alineados)."""
    if actual is None or "stay_date" not in fc.columns:
        return None
    work = fc.copy()
    work["stay_date"] = pd.to_datetime(work["stay_date"], errors="coerce")
    work["final_forecast_rn"] = pd.to_numeric(work.get("final_forecast_rn"), errors="coerce")
    work["_actual"] = pd.to_numeric(actual, errors="coerce").reindex(work.index)
    if work[["_actual", "final_forecast_rn"]].dropna().empty:
        return None
    daily = work.groupby("stay_date")[["final_forecast_rn", "_actual"]].sum().sort_index()

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(daily.index, daily["final_forecast_rn"], color=_COLORS["forecast"], lw=1.6, label="Forecast")
    ax.plot(daily.index, daily["_actual"], color=_COLORS["actual"], lw=1.4, label="Real")
    ax.set_title("Forecast vs Real · RN del portafolio", loc="left", fontsize=11)
    ax.set_ylabel("RN")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _scatter_forecast_vs_actual(fc: pd.DataFrame, actual: pd.Series | None) -> str | None:
    """Dispersión forecast vs real con diagonal de referencia."""
    if actual is None:
        return None
    f = pd.to_numeric(fc.get("final_forecast_rn"), errors="coerce")
    a = pd.to_numeric(actual, errors="coerce").reindex(fc.index)
    mask = f.notna() & a.notna()
    if mask.sum() == 0:
        return None
    x, y = a[mask], f[mask]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lim = max(x.max(), y.max()) * 1.05 or 1
    ax.plot([0, lim], [0, lim], ls="--", c=_COLORS["capacity"], lw=1, label="ideal")
    ax.scatter(x, y, s=12, alpha=0.5, c=_COLORS["forecast"])
    ax.set_xlabel("Real RN")
    ax.set_ylabel("Forecast RN")
    ax.set_title("Dispersión", loc="left", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _fig_to_base64(fig)


def _chart_occupancy(fc: pd.DataFrame) -> str | None:
    """Forecast contra el techo de capacidad (agregado diario)."""
    if not {"stay_date", "final_forecast_rn", "capacity"}.issubset(fc.columns):
        return None
    work = fc.copy()
    work["stay_date"] = pd.to_datetime(work["stay_date"], errors="coerce")
    work["final_forecast_rn"] = pd.to_numeric(work["final_forecast_rn"], errors="coerce")
    work["capacity"] = pd.to_numeric(work["capacity"], errors="coerce")
    if work["capacity"].notna().sum() == 0:
        return None
    daily = work.groupby("stay_date")[["final_forecast_rn", "capacity"]].sum().sort_index()

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(daily.index, daily["capacity"], color=_COLORS["capacity"], lw=1.4, ls="--", label="Capacidad")
    ax.plot(daily.index, daily["final_forecast_rn"], color=_COLORS["forecast"], lw=1.6, label="Forecast")
    ax.fill_between(daily.index, daily["final_forecast_rn"], daily["capacity"],
                    where=daily["final_forecast_rn"] > daily["capacity"],
                    color=_COLORS["actual"], alpha=0.15, label="excede techo")
    ax.set_title("Ocupación · forecast contra techo de capacidad", loc="left", fontsize=11)
    ax.set_ylabel("RN")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


# ------------------------------------------------------------------
# gráficos por propiedad (remediación de la vista: la operación es por hotel)
# ------------------------------------------------------------------


def _chart_forecast_by_property(
    fc: pd.DataFrame, actual: pd.Series | None, model_by_property: dict | None
) -> str | None:
    """Small multiples: P50 + banda P10/P90 + capacidad + reales por propiedad."""
    if "property" not in fc.columns or "stay_date" not in fc.columns:
        return None
    work = fc.copy()
    work["stay_date"] = pd.to_datetime(work["stay_date"], errors="coerce")
    work["final_forecast_rn"] = pd.to_numeric(work.get("final_forecast_rn"), errors="coerce")
    work["_actual"] = (
        pd.to_numeric(actual, errors="coerce").reindex(work.index)
        if actual is not None else float("nan")
    )
    props = sorted(work["property"].dropna().unique())
    if not props:
        return None
    ncols = 3 if len(props) > 3 else max(len(props), 1)
    nrows = math.ceil(len(props) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 2.9 * nrows), squeeze=False)
    for ax, prop in zip(axes.flat, props):
        g = work.loc[work["property"] == prop].sort_values("stay_date")
        model = (model_by_property or {}).get(str(prop))
        ax.set_title(str(prop) + (f" · {model}" if model else ""), loc="left", fontsize=9)
        if g["final_forecast_rn"].notna().sum() == 0:
            ax.axis("off")
            continue
        if {"p10_forecast", "p90_forecast"} <= set(g.columns):
            p10 = pd.to_numeric(g["p10_forecast"], errors="coerce")
            p90 = pd.to_numeric(g["p90_forecast"], errors="coerce")
            if p10.notna().any() and p90.notna().any():
                ax.fill_between(g["stay_date"], p10, p90, color=_COLORS["band"], alpha=0.5)
        ax.plot(g["stay_date"], g["final_forecast_rn"], color=_COLORS["forecast"], lw=1.4)
        if "capacity" in g.columns:
            cap = pd.to_numeric(g["capacity"], errors="coerce")
            if cap.notna().any():
                ax.plot(g["stay_date"], cap, color=_COLORS["capacity"], lw=1.0, ls="--")
        if g["_actual"].notna().any():
            ax.plot(g["stay_date"], g["_actual"], color=_COLORS["actual"], lw=1.1)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    for ax in axes.flat[len(props):]:
        ax.axis("off")
    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    return _fig_to_base64(fig)


def _wmape_color(value: float) -> str:
    """Semáforo de WMAPE: verde <15%, azul <30%, rojo ≥30%."""
    if value < 0.15:
        return _COLORS["champ"]
    if value < 0.30:
        return _COLORS["forecast"]
    return _COLORS["actual"]


def _chart_accuracy_by_property(breakdown: pd.DataFrame | None) -> str | None:
    """Barras horizontales de WMAPE por propiedad, anotadas con bias."""
    if breakdown is None or breakdown.empty or "wmape" not in breakdown.columns:
        return None
    b = breakdown.copy().dropna(subset=["wmape"])
    if b.empty:
        return None
    b = b.sort_values("wmape")
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(b) + 1.3))
    ax.barh(b["property"].astype(str), b["wmape"], color=[_wmape_color(v) for v in b["wmape"]])
    for y, (w, bias) in enumerate(zip(b["wmape"], b.get("bias", pd.Series(dtype=float)))):
        ax.text(w, y, f"  bias {bias:+.3f}", va="center", fontsize=7.5, color="#475569")
    ax.set_title("WMAPE por propiedad (menor = mejor; verde <15%, azul <30%)",
                 loc="left", fontsize=11)
    ax.set_xlabel("WMAPE")
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.25, axis="x")
    return _fig_to_base64(fig)


def _table_accuracy_by_property(
    breakdown: pd.DataFrame | None,
    model_by_property: dict | None,
    weekly_breakdown: pd.DataFrame | None = None,
    interval_coverage: dict | None = None,
    cold_start: dict[str, str] | None = None,
) -> str:
    """Tabla propiedad | modelo | WMAPE | WMAPE sem | MAPE | bias | cobertura | n.

    El MAPE diario lleva "†" cuando el decil inferior de los reales
    evaluados cae por debajo de 10 RN: con reales de pocas noches, ±5 RN
    de error ya son '+125%' — el MAPE diario es ruido de denominador,
    no señal de precisión (la semana mide la operación real).

    Con ``interval_coverage`` se agrega la cobertura empírica de la banda
    P10–P90 por propiedad (una banda honesta contiene el 80% nominal);
    con ``cold_start`` se marca ⚠ a las propiedades provisionales
    (< 365d de histórico).
    """
    if breakdown is None or breakdown.empty:
        return ""
    b = breakdown.copy().sort_values("wmape")
    weekly: dict[str, float] = {}
    weekly_note = False
    if weekly_breakdown is not None and not weekly_breakdown.empty:
        wb = weekly_breakdown
        if "property" in wb.columns and "wmape" in wb.columns:
            weekly = {
                str(p): float(w) for p, w in zip(wb["property"], wb["wmape"])
                if pd.notna(w)
            }
    cov_by_prop: dict[str, dict] = {}
    if interval_coverage:
        for row in interval_coverage.get("by_property") or []:
            cov_by_prop[str(row.get("property"))] = row
    cold = {str(p) for p in (cold_start or {})}
    rows = []
    for _, r in b.iterrows():
        prop = str(r.get("property"))
        model = (model_by_property or {}).get(prop, "—")
        prop_display = prop + ' <span class="flag">⚠ provisional</span>' if prop in cold else prop
        def _f(key: str, fmt: str) -> str:
            try:
                return format(float(r[key]), fmt)
            except (KeyError, TypeError, ValueError):
                return "—"
        mape_cell = _f("mape", ".1f")
        try:
            p10 = float(r["p10_actual"])
        except (KeyError, TypeError, ValueError):
            p10 = float("nan")
        small_denominator = bool(pd.notna(p10) and p10 < 10.0)
        if small_denominator:
            mape_cell += "†"
            weekly_note = True
        w_sem = weekly.get(prop)
        w_sem_cell = format(w_sem, ".3f") if w_sem is not None else "—"
        if w_sem is not None:
            weekly_note = True
        cov_cell = "—"
        cov_row = cov_by_prop.get(prop)
        if cov_row and int(cov_row.get("n_obs", 0)) > 0:
            cov = float(cov_row["coverage"])
            nominal = float(cov_row.get("nominal", 0.80))
            sev = "cov-ok" if abs(cov - nominal) <= 0.05 else (
                "cov-warn" if abs(cov - nominal) <= 0.10 else "cov-bad"
            )
            cov_cell = f'<span class="{sev}">{cov:.0%}</span>'
        rows.append(
            f"<tr><td>{prop_display}</td><td>{model}</td>"
            f"<td>{_f('wmape', '.3f')}</td><td>{w_sem_cell}</td>"
            f"<td>{mape_cell}</td>"
            f"<td>{_f('bias', '+.3f')}</td><td>{cov_cell}</td>"
            f"<td>{_f('n_obs', ',')}</td></tr>"
        )
    nota = ""
    if weekly_note:
        nota = (
            '<p style="font-size:0.78em;color:#64748b;margin:4px 0 0 0">'
            "WMAPE sem = precisión agregando forecast y reales por semana "
            "(grano operativo). †MAPE diario con reales diminutos "
            "(P10 < 10 RN): leer con el grano semanal. Cobertura = % de "
            "reales dentro de la banda P10–P90 (nominal 80%); verde dentro "
            "de ±5pp, ámbar ±10pp, rojo fuera. ⚠ provisional = < 365d de "
            "histórico.</p>"
        )
    return (
        '<table class="tbl"><thead><tr><th>Propiedad</th><th>Modelo</th>'
        "<th>WMAPE</th><th>WMAPE sem</th><th>MAPE</th><th>Bias</th>"
        "<th>Cobertura</th><th>n obs</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>" + nota
    )


def _chart_backtest_by_property(backtest_report: dict | None) -> str | None:
    """WMAPE de las revisiones T-90..T-1 por propiedad (campeón operante)."""
    if not backtest_report:
        return None
    rows = backtest_report.get("by_property") or []
    if not rows:
        return None
    df = pd.DataFrame(rows).dropna(subset=["wmape"]).sort_values("wmape")
    if df.empty:
        return None
    labels = [f"{r['property']} ({r.get('model', '?')})" for _, r in df.iterrows()]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(df) + 1.5))
    ax.barh(labels, df["wmape"], color=[_wmape_color(v) for v in df["wmape"]])
    for y, (w, n) in enumerate(zip(df["wmape"], df.get("n_obs", 0))):
        ax.text(w, y, f"  n={int(n)}", va="center", fontsize=7.5, color="#475569")
    ax.set_title("Revisiones T-90..T-1 · WMAPE por propiedad (backtest muestreado)",
                 loc="left", fontsize=11)
    ax.set_xlabel("WMAPE")
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.25, axis="x")
    return _fig_to_base64(fig)


def _chart_revisions_by_lead(backtest_report: dict | None) -> str | None:
    """Convergencia: WMAPE por banda de lead (una línea por propiedad)."""
    if not backtest_report:
        return None
    rows = backtest_report.get("by_property_lead_band") or []
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if not {"property", "lead_band", "wmape"} <= set(df.columns):
        return None
    order = ["90+", "60-89", "30-59", "14-29", "7-13", "3-6", "1-2"]
    present = [bnd for bnd in order if bnd in set(df["lead_band"])]
    if not present:
        return None
    x = range(len(present))
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for prop, g in df.groupby("property"):
        s = g.set_index("lead_band")["wmape"].reindex(present)
        ax.plot(x, s.values, marker="o", ms=3.5, lw=1.4, label=str(prop))
    ax.set_xticks(list(x))
    ax.set_xticklabels(present)
    ax.set_title("Convergencia · WMAPE por anticipación (T-90 → T-1, una línea por propiedad)",
                 loc="left", fontsize=11)
    ax.set_xlabel("Anticipación (banda de lead)")
    ax.set_ylabel("WMAPE")
    ax.legend(fontsize=7.5, ncol=min(3, df["property"].nunique()), loc="upper right")
    ax.grid(alpha=0.25)
    return _fig_to_base64(fig)


def _table_backtest_lead_band(backtest_report: dict | None) -> str:
    """Tabla global del backtest por banda de lead."""
    if not backtest_report:
        return ""
    rows = backtest_report.get("by_lead_band") or []
    if not rows:
        return ""
    order = ["90+", "60-89", "30-59", "14-29", "7-13", "3-6", "1-2"]
    df = pd.DataFrame(rows)
    present = [b for b in order if b in set(df.get("lead_band", []))]
    if present:
        df = df.set_index("lead_band").reindex(present).reset_index()
    cells = []
    for _, r in df.iterrows():
        def _f(key: str, fmt: str) -> str:
            try:
                return format(float(r[key]), fmt)
            except (KeyError, TypeError, ValueError):
                return "—"
        cells.append(
            f"<tr><td>{r.get('lead_band', '—')}</td><td>{_f('n_obs', ',')}</td>"
            f"<td>{_f('wmape', '.3f')}</td><td>{_f('mape', '.1f')}</td></tr>"
        )
    return (
        '<table class="tbl"><thead><tr><th>Banda</th><th>n obs</th>'
        "<th>WMAPE</th><th>MAPE</th></tr></thead><tbody>"
        + "".join(cells) + "</tbody></table>"
    )


# ------------------------------------------------------------------
# ensamblado HTML
# ------------------------------------------------------------------


def _img_tag(b64: str | None, alt: str) -> str:
    if not b64:
        return f'<p class="muted">— {alt}: sin datos en esta corrida.</p>'
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}"/>'


def _kpi_grid(kpis: dict | None, champion: str | None = None) -> str:
    """Franja de KPIs ejecutivos: etiqueta ceja + valor display serif.

    Los valores usan tipografía serif editorial con dígitos tabulares — el
    lenguaje visual de un reporte financiero, no de un dashboard genérico.
    """
    if not kpis and not champion:
        return ""
    labels = {
        "forecast_room_nights": "Forecast RN",
        "forecast_occupancy": "Ocupación",
        "forecast_bias": "Bias",
        "expected_pickup": "Pickup esperado",
        "capacity_utilization": "Uso capacidad",
        "n_properties_evaluated": "Propiedades evaluadas",
        "forecast_accuracy_wmape": "WMAPE",
        "forecast_accuracy_mape": "MAPE",
    }
    cells = []
    if champion:
        cells.append(
            f'<div class="kpi"><span class="kpi-l">Campeón</span>'
            f'<span class="kpi-v kpi-model">{champion}</span>'
            f'<span class="kpi-x">modelo operante</span></div>'
        )
    for key, label in labels.items():
        if not kpis or key not in kpis:
            continue
        val = kpis[key]
        try:
            val = float(val)
            if key == "n_properties_evaluated":
                total = float(kpis.get("n_properties_portfolio", 0) or 0)
                shown = f"{val:.0f} / {total:.0f}"
            elif "occupancy" in key or "utilization" in key:
                shown = f"{val:.1%}"
            elif key == "forecast_accuracy_mape":
                shown = f"{val:.1f}%"
            elif key == "forecast_accuracy_wmape":
                shown = f"{val:.3f}"
            elif key == "forecast_bias":
                shown = f"{val:+.1%}"
            else:
                shown = f"{val:,.1f}"
        except (TypeError, ValueError):
            shown = str(val)
        hint = {
            "forecast_accuracy_wmape": "ponderado por volumen",
            "forecast_accuracy_mape": "promedio simple por día",
            "forecast_bias": "negativo = sub-estima",
            "forecast_room_nights": "noches en la ventana",
            "forecast_occupancy": "vs inventario disponible",
            "expected_pickup": "además de lo reservado",
            "capacity_utilization": "inventario usado",
            "n_properties_evaluated": "con reales en la ventana",
        }.get(key, "")
        cells.append(
            f'<div class="kpi"><span class="kpi-l">{label}</span>'
            f'<span class="kpi-v">{shown}</span>'
            f'{f"<span class=kpi-x>{hint}</span>" if hint else ""}</div>'
        )
    if not cells:
        return ""
    return '<div class="kpis">' + "".join(cells) + "</div>"


def _fmt_metrics(metrics: dict | None) -> str:
    if not metrics:
        return '<p class="muted">Sin métricas en esta corrida.</p>'
    rows = []
    for k, v in metrics.items():
        try:
            rows.append(f"<tr><td>{k}</td><td>{float(v):,.4f}</td></tr>")
        except (TypeError, ValueError):
            rows.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    return ('<table class="tbl"><thead><tr><th>Métrica</th><th>Valor</th></tr></thead>'
            "<tbody>" + "".join(rows) + "</tbody></table>")


def generate_visual_report(
    run_dir: str | Path,
    forecast: pd.DataFrame,
    *,
    demand: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
    metrics: dict | None = None,
    kpis: dict | None = None,
    actual: pd.Series | None = None,
    property_breakdown: pd.DataFrame | None = None,
    property_weekly_breakdown: pd.DataFrame | None = None,
    model_by_property: dict | None = None,
    backtest_report: dict | None = None,
    champion: str | None = None,
    significance: dict | None = None,
    interval_coverage: dict | None = None,
    cold_start: dict[str, str] | None = None,
    drift_alerts: list | None = None,
    monitoring_state: dict | None = None,
    bias_correction: dict | None = None,
    vintage_tracking: dict | None = None,
    forecast_revision: dict | None = None,
    lead_band_breakdown: list[dict] | None = None,
    benchmark_bands: dict | None = None,
    tracking_signal_alert: float | None = None,
    track_record: list[dict] | None = None,
    run_id: str | None = None,
    run_mode: str | None = None,
    title: str = "Room Nights Forecast Engine",
) -> Path:
    """Escribe ``<run_dir>/report.html`` con los gráficos embebidos.

    Todos los parámetros de datos son opcionales: las secciones ausentes se
    omiten con un mensaje en lugar de fallar la corrida. El detalle por
    propiedad (forecast/bandas/modelo, accuracy y revisiones del backtest
    T-90..T-1) se activa con ``property_breakdown``/``model_by_property``/
    ``backtest_report``/``champion``; ``property_weekly_breakdown`` agrega
    la columna de WMAPE semanal a la tabla de accuracy; sin ellos el
    reporte queda como la vista de portafolio clásica.

    Auto-explicado (spec reporting): cada sección lleva su guía de lectura
    plegable, un glosario arriba y un panel de **interpretación y
    validación** con veredictos por métrica (WMAPE, bias, cobertura de
    banda, convergencia, significancia del campeón, cold start, drift YoY
    y estado del registro). Los veredictos son ayuda de lectura — los
    gates duros viven en los criterios de aceptación del runner.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # sistema visual: los gráficos hablan el mismo idioma que la página
    _apply_chart_style()

    forecast = _with_lead_band(forecast)
    actual_series = actual
    if actual_series is not None:
        actual_series = pd.Series(actual_series).reset_index(drop=True)
        if len(actual_series) != len(forecast):
            actual_series = None

    def _card(name: str, img: str | None, extra: str = "",
              wide: bool = False, placeholder: bool = True) -> str:
        """Tarjeta de sección.

        ``placeholder=True``: sin datos → la sección aparece con un mensaje
        (el lector sabe que existe y por qué está vacía). ``placeholder=False``:
        sin imagen NI extra → la sección se omite (secciones que dependen de
        desgloses opcionales: vs-real, accuracy, revisiones, dispersión).
        """
        if img is None and not extra and not placeholder:
            return ""
        img_html = _img_tag(img, name) if img else (
            f'<p class="muted">— {name}: sin datos en esta corrida.</p>'
            if placeholder else ""
        )
        cls = "card wide" if wide else "card"
        return (
            f'<section class="{cls}"><h2>{name}</h2>'
            f"{img_html}{extra}{_explainer(name)}</section>"
        )

    # ---- resumen ejecutivo: el tablero de veredictos va primero
    resumen = _validation_panel(
        metrics=metrics,
        benchmark=benchmark,
        champion=champion,
        significance=significance,
        interval_coverage=interval_coverage,
        backtest_report=backtest_report,
        cold_start=cold_start,
        drift_alerts=drift_alerts,
        monitoring_state=monitoring_state,
        bias_correction=bias_correction,
        vintage_tracking=vintage_tracking,
        forecast_revision=forecast_revision,
        lead_band_breakdown=lead_band_breakdown,
        benchmark_bands=benchmark_bands,
        tracking_signal_alert=tracking_signal_alert,
        track_record=track_record,
    )

    # ---- precisión: cómo de bien pronostica el motor
    rev_lead_img = _chart_revisions_by_lead(backtest_report)
    rev_extra = ""
    if rev_lead_img:
        rev_extra += _img_tag(rev_lead_img, "convergencia por anticipación")
    rev_extra += _table_backtest_lead_band(backtest_report)

    precision = [
        _card("Benchmark", _chart_benchmark(benchmark, champion)),
        _card(
            "Accuracy por propiedad",
            _chart_accuracy_by_property(property_breakdown),
            extra=_table_accuracy_by_property(
                property_breakdown, model_by_property, property_weekly_breakdown,
                interval_coverage=interval_coverage, cold_start=cold_start,
            ),
            wide=True,
            placeholder=False,
        ),
        _card(
            "Revisiones T-90..T-1 · por propiedad",
            _chart_backtest_by_property(backtest_report),
            extra=rev_extra,
            wide=True,
            placeholder=False,
        ),
        _card("Dispersión", _scatter_forecast_vs_actual(forecast, actual_series),
              placeholder=False),
        *(
            [_card(
                "Expediente del forecast publicado",
                _chart_track_record(track_record),
                extra=_table_track_record(track_record) + _explainer_track_record(),
                wide=True,
                placeholder=False,
            )] if track_record else []
        ),
    ]

    # ---- pronóstico: lo que se publica y de qué está hecho
    forecast_cards = [
        _card("Forecast", _chart_forecast(forecast), wide=True),
        _card(
            "Forecast vs Real",
            _chart_forecast_vs_actual(forecast, actual_series),
            wide=True,
            placeholder=False,
        ),
        _card("Forecast por propiedad",
              _chart_forecast_by_property(forecast, actual_series, model_by_property),
              wide=True),
        _card("Ocupación", _chart_occupancy(forecast)),
        _card("Pickup", _chart_pickup(forecast)),
    ]

    # ---- contexto: el mundo sobre el que se entrena y el vocabulario
    contexto = [
        _card("EDA", _chart_eda(demand), wide=True),
        (
            '<section class="card"><h2>Métricas</h2>'
            f"{_fmt_metrics(metrics)}</section>"
        ),
        _glossary_card(),
    ]

    def _group(eyebrow: str, cards: list[str]) -> str:
        clean = [c for c in cards if c]
        if not clean:
            return ""
        return (
            f'<div class="group"><p class="eyebrow">{eyebrow}</p>'
            f'<div class="grid">{"".join(clean)}</div></div>'
        )

    groups = "".join([
        _group("Resumen ejecutivo", [resumen]),
        _group("Precisión", precision),
        _group("Pronóstico", forecast_cards),
        _group("Contexto", contexto),
    ])

    # ---- masthead: la franja de identidad + metadatos de la corrida
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta: list[str] = []
    if run_id:
        meta.append(f'<span class="meta"><b>corrida</b>{run_id}</span>')
    if run_mode:
        meta.append(f'<span class="meta"><b>modo</b>{run_mode}</span>')
    if "stay_date" in forecast.columns:
        stays = pd.to_datetime(forecast["stay_date"], errors="coerce").dropna()
        if not stays.empty:
            meta.append(
                f'<span class="meta"><b>ventana</b>'
                f"{stays.min().date()} → {stays.max().date()}</span>"
            )
    if champion:
        meta.append(f'<span class="meta"><b>campeón</b>{champion}</span>')
    meta.append(f'<span class="meta"><b>generado</b>{stamp}</span>')
    meta_html = f'<div class="mast-meta">{"".join(meta)}</div>' if meta else ""

    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title} · {run_id or 'corrida'}</title>
<style>
  :root {{
    color-scheme: light;
    --paper: #F4F7F9; --card: #FFFFFF;
    --ink: #0B2239; --text: #334155; --muted: #64748B; --faint: #94A3B8;
    --line: #E2E8F0; --line-soft: #EEF2F6;
    --petrol: #0E7490; --petrol-deep: #155E75; --petrol-tint: #E0F2F7;
    --green: #15803D; --green-tint: #ECFDF5;
    --amber: #B45309; --amber-tint: #FEF6E7;
    --red: #B91C1C; --red-tint: #FDEEEE;
    --serif: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    --mono: ui-monospace, "Cascadia Mono", Consolas, "Courier New", monospace;
    --sans: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--paper); color: var(--text);
          font-family: var(--sans); font-size: 14px; line-height: 1.5; }}
  /* ---- masthead: membrete de la corrida (la firma del reporte) ---- */
  .mast {{ background: var(--card); border-top: 3px solid var(--petrol);
           border-bottom: 1px solid var(--line); }}
  .mast-inner {{ max-width: 1160px; margin: 0 auto; padding: 18px 28px 14px;
                 display: flex; flex-wrap: wrap; gap: 12px 32px;
                 justify-content: space-between; align-items: flex-end; }}
  .mast-brand {{ display: flex; align-items: center; gap: 14px; }}
  .mark {{ width: 40px; height: 40px; border-radius: 8px;
           background: var(--ink); color: #fff; font: 700 15px var(--mono);
           display: flex; align-items: center; justify-content: center;
           letter-spacing: .04em; }}
  .mast h1 {{ margin: 0; font-size: 19px; font-weight: 650; color: var(--ink);
              letter-spacing: -.01em; }}
  .mast-sub {{ margin: 2px 0 0; font-size: 12px; color: var(--muted); }}
  .mast-meta {{ display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: center; }}
  .meta {{ font: 400 11.5px var(--mono); color: var(--text); white-space: nowrap; }}
  .meta b {{ color: var(--faint); font-weight: 500; margin-right: 6px;
             text-transform: lowercase; }}
  main {{ padding: 26px 28px 10px; max-width: 1160px; margin: 0 auto; }}
  /* ---- KPIs: números serif con dígitos tabulares ---- */
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
           gap: 12px; margin-bottom: 26px; }}
  .kpi {{ background: var(--card); border: 1px solid var(--line);
          border-radius: 10px; padding: 12px 16px 11px; }}
  .kpi-l {{ display: block; font-size: 10.5px; font-weight: 600; color: var(--muted);
            text-transform: uppercase; letter-spacing: .08em; }}
  .kpi-v {{ display: block; font-family: var(--serif); font-size: 25px;
            font-weight: 600; color: var(--ink); margin-top: 3px;
            font-variant-numeric: tabular-nums; letter-spacing: -.01em; }}
  .kpi-v.kpi-model {{ font-family: var(--mono); font-size: 15px; color: var(--petrol-deep);
                      padding-top: 6px; }}
  .kpi-x {{ display: block; font-size: 11px; color: var(--faint); margin-top: 2px; }}
  /* ---- grupos: la estructura codifica el orden de lectura ---- */
  .group {{ margin-bottom: 26px; }}
  .eyebrow {{ margin: 0 0 10px; font-size: 11px; font-weight: 700;
              color: var(--petrol-deep); text-transform: uppercase;
              letter-spacing: .12em; display: flex; align-items: center; gap: 12px; }}
  .eyebrow::after {{ content: ""; flex: 1; height: 1px; background: var(--line); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr));
           gap: 16px; }}
  .card {{ background: var(--card); border: 1px solid var(--line);
           border-radius: 10px; padding: 16px 18px; }}
  .card.wide {{ grid-column: 1 / -1; }}
  .card h2 {{ margin: 0 0 10px; font-size: 13px; font-weight: 650; color: var(--ink);
              text-transform: uppercase; letter-spacing: .05em; }}
  .card img {{ width: 100%; height: auto; }}
  .card img + img {{ margin-top: 10px; }}
  /* ---- tablas ---- */
  table.tbl {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  table.tbl th {{ text-align: left; padding: 7px 10px 5px; color: var(--muted);
                  font-size: 10.5px; font-weight: 600; text-transform: uppercase;
                  letter-spacing: .06em; border-bottom: 1px solid var(--line); }}
  table.tbl td {{ text-align: left; padding: 7px 10px;
                  border-bottom: 1px solid var(--line-soft);
                  font-variant-numeric: tabular-nums; }}
  table.tbl tr:last-child td {{ border-bottom: none; }}
  .muted {{ color: var(--faint); font-style: italic; }}
  /* ---- guías de lectura plegables ---- */
  details.expl {{ margin-top: 12px; border-top: 1px dashed var(--line);
                  padding-top: 8px; }}
  details.expl summary {{ cursor: pointer; font-size: 12px; font-weight: 600;
                          color: var(--petrol-deep); }}
  details.expl summary:hover {{ color: var(--petrol); }}
  details.expl p {{ font-size: 12.5px; color: #475569; line-height: 1.6;
                    margin: 8px 0 0; }}
  /* ---- lámparas de veredicto ---- */
  .chip {{ display: inline-block; position: relative; padding: 3px 10px 3px 22px;
           font-size: 11.5px; font-weight: 600; border-radius: 999px;
           white-space: nowrap; }}
  .chip::before {{ content: ""; position: absolute; left: 9px; top: 50%;
                   transform: translateY(-50%); width: 7px; height: 7px;
                   border-radius: 50%; background: currentColor; }}
  .chip.ok {{ background: var(--green-tint); color: var(--green); }}
  .chip.warn {{ background: var(--amber-tint); color: var(--amber); }}
  .chip.bad {{ background: var(--red-tint); color: var(--red); }}
  .chip.info {{ background: var(--line-soft); color: var(--text); }}
  table.verdict th:nth-child(3) {{ width: 44%; }}
  table.verdict td {{ vertical-align: top; }}
  .flag {{ background: var(--amber-tint); color: var(--amber); font-size: 10.5px;
           font-weight: 600; border-radius: 5px; padding: 1px 6px;
           white-space: nowrap; }}
  .cov-ok {{ color: var(--green); font-weight: 600; }}
  .cov-warn {{ color: var(--amber); font-weight: 600; }}
  .cov-bad {{ color: var(--red); font-weight: 600; }}
  footer {{ color: var(--faint); font-size: 11.5px; padding: 8px 28px 34px;
            text-align: center; max-width: 1160px; margin: 0 auto; }}
  footer b {{ color: var(--muted); font-weight: 600; }}
  @media print {{
    body {{ background: #fff; }}
    .card, .kpi {{ break-inside: avoid; }}
    .grid {{ grid-template-columns: 1fr; }}
  }}
</style></head>
<body>
<div class="mast"><div class="mast-inner">
  <div class="mast-brand">
    <span class="mark">RN</span>
    <div>
      <h1>{title}</h1>
      <p class="mast-sub">Enjoy Costa Rica · Revenue Management · reporte de revisión</p>
    </div>
  </div>
  {meta_html}
</div></div>
<main>
  {_kpi_grid(kpis, champion)}
  {groups}
</main>
<footer><b>Reproducible</b>: misma seed → mismos resultados · los 10 entregables y el manifiesto quedan en la raíz de la corrida · vista generada automáticamente, la lógica vive en <b>src/</b></footer>
</body></html>"""

    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


__all__ = ["generate_visual_report"]
