# Design — add-holiday-features

## Context

La curva estacional (cambio anterior) dejó el 60–89 en ~38–40% MAPE. El residuo de señal disponible sin datos externos son los feriados: picos de demanda conocidos con años de anticipación (exógenos, sin leakage). El spec de features ya pedía "indicador de feriado cuando exista información confiable" — este cambio provee la fuente.

## Goals / Non-Goals

**Goals** — Calendario CR+US confiable (movibles incluidos) como primitiva; condicionamiento por feriado en la curva del campeón con la cadena de fallback ya probada; features de feriado para los GBM (bake-off parejo); defaults por A/B.

**Non-Goals** — Eventos privados (conciertos, congresos) fuera de alcance (requieren fuente externa); ADR/revenue; rediseño de selección.

## Decisions

### D1 — Fuente: librería `holidays` (CR + US), agrupación semántica
`src/features/holidays.py` construye, para un rango de años, las fechas de ambos países y las agrupa en claves semánticas estables: `semana_santa` (Jueves–Domingo Santo se funden), `fin_de_ano` (Navidad + Año Nuevo), `thanksgiving`, `memorial_day`, `labor_day_us`, `independencia_us`, `independencia_cr`, `anexion`, `dia_de_la_madre`, y slug genérico para el resto. La librería resuelve los movibles (computus para Semana Santa; N-ésimo día de semana para Thanksgiving/Memorial/Labor). Alternativa descartada: hand-roll del computus — más código propietario por la misma corrección.

### D2 — Ventanas de viaje y estructura del calendario
`HolidayCalendar` expone: `dates` (fechas exactas de feriado — para `is_holiday` y `days_to_nearest_holiday`) y `windows` (Series fecha→clave para cada día dentro de ±`window_days`; empates por el feriado más cercano). Default `window_days: 4`: Semana Santa cubre ~11 días (Jueves y Viernes Santo generan ventanas solapadas) y Navidad+Año Nuevo encadenan 21-dic→5-ene casi continuo. El calendario es una primitiva compartida: runner, benchmark y backtest reciben la MISMA instancia (cache por años+países+ventana).

### D3 — Cadena de condicionamiento: feriado → temporada → propiedad → global
`PickupCurveModel` gana `holiday_conditional` + `holiday_calendar`. El bucket (propiedad, feriado) se usa solo si acumula `min_stays_per_bucket` estancias (mismo umbral; un bucket delgado cae a temporada — sin daño). Ambos condicionamientos respetan `season_min_lead`: a leads cortos el on-the-books domina y el pooling por propiedad es menos ruidoso que cualquier bucket fino. Alternativa descartada: buckets (temporada, feriado) anidados — fragmentación insostenible con ~1000–4000 estancias por propiedad.

### D4 — Bake-off parejo: feriados también para los GBM
`ML_FEATURES` gana `days_to_nearest_holiday` (continua; los árboles hacen splits tipo |d−feriado| ≤ k). Sin esto, dar señal de feriados solo al campeón sesga el benchmark a favor de pickup_curve. `is_holiday` (fecha exacta) se agrega al frame de features de la corrida vía `add_calendar_features(holidays=...)` — la API ya existía.

### D5 — Calendario viaja explícito, no por config-JSON
`fit_candidate`/`build_forecasts`/`real_fold_metrics` reciben `holiday_calendar` como argumento separado de `curve_params` (un objeto no es serializable en el manifiesto); `config.holidays` solo define países/ventana/umbral y el runner construye la instancia. El manifiesto registra la configuración (países, ventana, `holiday_conditional`) sin el objeto.

### D6 — Validación: A/B estándar + backtest dirigido (resultados y decisión)
**Estandar** (fit al corte 2026-07-08, ventana Jul–Oct, 6 propiedades):
leads 1–29 idénticos (el umbral los preserva), 30–59: 0.1867→0.1861,
60–89: 0.2319→0.2265; global WMAPE 0.1783→**0.1764**, MAPE 27.53→**26.79%**.
Sin regresión por propiedad; LIREL mejora 0.1875→0.1683.

**Dirigido** (ventanas grandes con reales, fit ~60d antes, lead 60–69):

| Ventana | F-temp | H15 |
|---|---|---|
| semana_santa_2026 | 21.2% | **19.9%** |
| semana_santa_2025 | 26.0% | 26.0% |
| fin_de_ano_2025 | 21.8% | **21.0%** |
| thanksgiving_2025 | 23.9% | 23.9% |
| independencia_cr_2025 | 23.6% | 23.6% |
| dia_madre_2026 | 36.2% | **27.9%** |
| **Total (WMAPE)** | 0.2075 | **0.2001** |

**Decisión**: `holiday_conditional: true`, `window_days: 4`,
`min_stays_per_holiday_bucket: 15`. Hallazgo estructural del experimento: con
el mismo umbral de las temporadas (30 estancias) el nivel de feriado es
**inerte** (solo 3 buckets en todo el portafolio: una ventana de ±4 días × 3
años de recencia ≈ 27 estancias, por debajo de la barra) — de ahí el umbral
propio `min_stays_per_holiday_bucket` (None ⇒ reutiliza el de temporadas).
**Caveats honestos**: la ganancia está concentrada en las ventanas con
bucket activo (Semana Santa 2026, Día de la Madre, fin de año); Semana Santa
2025 y Thanksgiving 2025 quedaron idénticas (buckets aún delgados en esos
cortes); el umbral de 15 estancias roza el ruido, pero la cadena de fallback
+ el gate de lead garantizan ausencia de daño (cero regresiones observadas
en ambos experimentos).

## Risks / Trade-offs

- [Buckets de feriado ruidosos en propiedades chicas] → `min_stays_per_bucket` + fallback a temporada; los feriados medianos con efecto nulo simplemente replican la media de temporada.
- [Feriados CR menores (Juan Santamaría, Trabajo) sin efecto hotelero real] → quedan como slugs propios; sin evidencia suficiente no generan bucket.
- [A/B de ventana con feriados medianos puede no detectar el efecto] → backtest dirigido a las ventanas grandes como evidencia complementaria; decisión documentada con ambos.
- [Runtime: `days_to_nearest_holiday` por fila ML] → searchsorted sobre `dates` ordenado; costo trivial.

## Migration Plan

1. `holidays` a requirements + módulo calendario → modelo → wiring → tests → A/B (estándar + dirigido) → default → corridas → archive. Rollback: `holidays.countries: []` desactiva todo el feature por YAML.
