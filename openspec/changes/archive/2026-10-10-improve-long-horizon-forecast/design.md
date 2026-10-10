# Design — improve-long-horizon-forecast

## Context

El campeón operativo (`pickup_curve`) calcula `forecast = booked + E[remaining pickup | lead]` con la media por bin de lead entrenada sobre TODA la historia y sin distinguir la temporada del stay. A leads 60–90 el término `booked` es pequeño, así que el forecast converge al promedio anual de pickup restante — exactamente donde el error del horizonte se dispara (MAPE 42.8% en 60–89). Simultáneamente LIRAK y LAPAS están cerradas: sus datos rancios/huecos contaminan benchmark, backtest y KPIs del portafolio.

## Goals / Non-Goals

**Goals**
- Exclusión configurable de propiedades cerradas, propagada a toda la corrida y registrada.
- Curva de pickup condicionada por temporada del stay con ventana de recencia y fallback por escasez.
- Defaults fijados por evidencia (A/B por banda de lead sobre la ventana de verificación + backtest por lead), no por intuición.

**Non-Goals**
- Cambiar la arquitectura champion/challenger ni el mecanismo de selección.
- Features externas (feriados/eventos/ADR/clima): siguiente cambio, no este.
- Tuning de los candidatos ML para largo plazo.

## Decisions

### D1 — Exclusión en la ingesta, tras alias, antes de calidad
El filtro `data.excluded_properties` se aplica a las tres fuentes crudas justo después de resolver alias (las cerradas se nombran por su nombre canónico: LIRAK, LAPAS). Al filtrar antes de `raw_sources`, la calidad, features, benchmark, forecast, backtest y monitoring quedan limpios por construcción — un solo punto de exclusión en cada entrada (`load_real_sources` y `run_end_to_end`). El manifiesto registra `excluded_properties` con motivo "propiedad cerrada (configuración)"; el coverage report las separa de las "sin reales en la ventana".
Alternativa descartada: excluir solo en el forecast — dejaría que benchmark/backtest/KPIs sigan contaminados.

### D2 — Condicionamiento por temporada en el MISMO candidato
`PickupCurveModel` gana tres parámetros: `season_conditional`, `max_history_days`, `min_stays_per_bucket`. Con `season_conditional=True` se aprende una curva por (propiedad, temporada del stay — high/shoulder/low, mismo mapa que `features/calendar`) además de la curva por propiedad y la global; el lookup degrada (propiedad, temporada) → propiedad → global cuando el bucket no acumula `min_stays_per_bucket` estancias. `predict` ya recibe `stay_date`, así que la temporada del stay está disponible sin cambios de interfaz. El candidato sigue llamándose `pickup_curve`: es el mismo modelo, ahora con la no-linealidad estacional que la spec feature-engineering ya exigía ("por… mes, temporada").
Alternativa descartada: un candidato nuevo `pickup_curve_seasonal` — inflaría el tablero con una variante que es el mismo modelo con parámetros; el benchmark mediría dos configs del mismo método en vez de métodos distintos.

### D3 — Ventana de recencia relativa a la última estancia de la curva
`max_history_days` (default 1095) recorta las estancias de entrenamiento a `max(stay_date) − ventana`. Amortigua regímenes viejos y alinea la curva con el comportamiento reciente de reserva; con buckets estacionales de 3 temporadas, 3 años de stays dejan ~250–900 estancias por temporada en las propiedades activas — suficiente por encima del umbral de fallback.
Alternativa descartada: ponderación exponencial por recencia — más parámetros, mismo efecto práctico con estos volúmenes.

### D4 — Defaults por experimento A/B, no por corazonada
Protocolo ejecutado (fit al corte 2026-07-08, 6 propiedades activas, ventana
Jul–Oct con reales, comparación por banda de lead):

| Variante | Global WMAPE | 1-6d | 14-29 | 30-59 | 60-89 |
|---|---|---|---|---|---|
| A estática (todas) | 0.1864 | 0.049 | 0.146 | 0.193 | 0.251 |
| D solo-recencia | 0.1864 | 0.049 | 0.146 | 0.193 | 0.251 |
| C/E temporada (+recencia) | 0.1786 | 0.056 | 0.144 | 0.187 | 0.232 |
| **F temporada + `season_min_lead: 30`** | **0.1783** | **0.049** | 0.146 | **0.187** | **0.232** |

Decisiones tomadas de la evidencia:
- **Recencia es inerte en estos datos** (D ≡ A exacto: el panel de pickup no
  tiene historia fuera de la ventana) — se conserva como knob protector de
  regímenes futuros, no como mejora actual.
- **Temporada sola degrada leads 1–6** (0.049→0.056): a leads cortos el
  bucket estacional tiene menos datos por bin que el pooling de propiedad.
  → nuevo parámetro `season_min_lead` (default 30): la curva estacional
  aplica solo a leads ≥ 30; debajo, la curva agrupada.
- **Se adopta la variante F**: ganancia íntegra en 30–89 (60-89: 0.251→0.232,
  −7.8% relativo) con leads 1–29 preservados exactamente.
- **Heterogeneidad por propiedad declarada como watch item**: VILLAS mejora
  drásticamente (0.265→0.160, corrige su sesgo +18.5%) y CORIN mejora; LIREL
  (0.150→0.184) y MARINA (0.197→0.258) empeoran con la curva estacional en
  esta ventana. Con una sola ventana de evaluación no se configuran curvas
  por propiedad (sobre-ajuste); el benchmark trimestral continuo aporta
  evidencia para revisarlo.

## Risks / Trade-offs

- [Bucket estacional ruidoso en propiedad chica → forecast peor que la curva global] → `min_stays_per_bucket` (default 30) con fallback determinístico; SJOSL esperablemente cae al fallback.
- [La ventana de recencia descarta señal en propiedades con historia corta] → la ventana es un máximo: propiedades con menos de 1095d usan toda su historia.
- [Menos propiedades en el portafolio → KPIs no comparables con corridas viejas] → el manifiesto declara la exclusión; las comparaciones entre corridas deben leer el `excluded_properties`.
- [Sobre-ajuste del A/B a una sola ventana] → el A/B decide defaults, no declara la verdad; la evidencia continua viene del backtest y de futuras corridas de verificación.

## Migration Plan

1. Config + modelo + wiring → tests unitarios → A/B → defaults → corrida de verificación y futuro → suite completa.
2. Si el A/B muestra regresión en leads cortos: `season_conditional: false` queda como config (el cambio de behavior es reversible por YAML).
3. Archivo del cambio OpenSpec con sync de specs.
