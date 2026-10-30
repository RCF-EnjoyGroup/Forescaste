# Proposal — seasonal-calibration-cells

## Contexto (evidencia real-20261030-1147, reales al 29-oct)

La banda publicada volvió a descalibrar en la misma dirección que lleva 4
corridas — pero ahora con la raíz completamente diagnosticada:

- **FIESTA: cobertura 42.2% (n=90, nominal 80%)**. De los 90 días del
  horizonte (ago-oct), **51 reales EXCEDEN el p90 y solo 1 cae por debajo
  del p10** — la banda no es inestable: está anclada a un punto
  sistemáticamente bajo (p50 medio 312.9 vs actual medio 361.6 = **−13.5%**
  de sesgo de punto en el régimen del horizonte).
- La corrección de bias de punto (p50) **no se aplicó**: el gate la
  rechazó en las bandas grandes (30-59, 60-89, 14-29) porque el sesgo
  entre folds tiene **consistencia 50% < 75%** — el signo del sesgo gira
  entre ventanas. Es decir: el mismo modelo que hoy pronostica 13.5% abajo
  en el régimen ago-oct, en otros folds del año pronosticó arriba; la media
  del sesgo es un promedio de regímenes opuestos, no un bias.
- En paralelo **LIREL (96.7%) y VILLAS (94.4%) sobre-cubren** — bandas más
  anchas que el error real de su régimen (amplitud media LIREL 31.6 RN
  sobre reales medios de 46.6).

## Causa raíz (la misma para ambas direcciones)

La calibración de la banda y la corrección de bias se estiman por
**(propiedad × banda de lead)** a lo largo de un año de folds que mezcla
todas las estaciones. El horizonte vive en UNA temporada. El resultado:

1. El sesgo por celda (prop × banda) oscila con la estación → el gate de
   consistencia bloquea (correctamente: sesgo que gira es régimen, no
   bias) → el punto sale sin corrección → banda anclada baja → fallos
   encimados.
2. Los cuantiles de residuos mezclan regímenes → amplitud inflada en
   propiedades cuyo régimen del horizonte es más estable que el promedio
   anual → sobre-cobertura (LIREL, VILLAS).

El faltante es una dimensión conocible a-priori (el calendario): la
**temporada del stay**. «Lead 30-59 en temporada baja» es una celda
coherente; «lead 30-59 en todas las estaciones» es un promedio de
regímenes opuestos.

## Propuesta

Extender las celdas de calibración (bias del punto y amplitud de la banda)
de **(propiedad, banda de lead)** a **(propiedad, banda de lead, temporada
del stay)** `high/shoulder/low` (misma definición del desglose de
estacionalidad que ya usa el reporte), manteniendo TODO el andamiaje actual:

- mismos gates: `min_obs=20`, consistencia de signo entre folds, shrinkage
  `n/(n+30)`, clip ±30%;
- escalera de fallback de la banda: celda estacional → (prop, banda) →
  propiedad → banda → global (documentada en `note`);
- la corrección del punto con fallback honesto: si la celda estacional no
  alcanza la evidencia, se conserva la corrección de la celda (prop ×
  banda) de siempre;
- la banda siempre calibra el pipeline completo publicado (punto
  corregido + banda) con cuantiles completos del residuo.

Bajo knob `bias_correction.seasonal_cells` — el default se decide con la
validación A/B, no a priori.

## Validación A/B — RESULTADO NEGATIVO (decisión documentada)

A/B ejecutado el 30-oct con data/seed idénticos:
baseline `real-20261030-1147` (knob off) vs `real-20261030-1253` (knob on).

| Propiedad | Cobertura off | Cobertura on | Δ | WMAPE off → on |
|---|---|---|---|---|
| CORIN | 72.2% | 60.0% | **−12.2pp** | 21.6 → 22.2% |
| FIESTA | 42.2% | 23.3% | **−18.9pp** | 14.7 → 15.9% |
| LIREL | 96.7% | 95.6% | −1.1pp | 14.3 → 14.2% |
| MARINA | 66.7% | 64.4% | −2.2pp | 37.7 → 38.5% |
| SJOSL | 67.8% | 67.8% | 0.0 | 42.9 → 42.9% |
| VILLAS | 94.4% | 91.1% | −3.3pp | 25.1 → **22.3%** |

**El mecanismo SÍ funcionó** (43 celdas estacionales con fuente aplicada
en el manifiesto; punto y banda recalculados fila a fila). Lo que DON'T
funcionó es la hipótesis: el sesgo −13.5% de FIESTA en el horizonte es
**drift de nivel YoY** — una anomalía del año en curso que NO existe en la
data de calibración del año anterior. Los factores estacionales del año
previo dijeron «corrige a la baja» (el año pasado sobre-pronosticó en esa
temporada) — la dirección exactamente opuesta al boom actual — y el punto
saló más bajo (p50 medio 312.9 → 307.6) con banda más estrecha
(p90 medio 352.5 → 339.6): la cobertura se derrumbó.

**Decisión: `seasonal_cells: false` por default.** La maquinaria queda
disponible como knob con la evidencia documentada en el config (precedente:
`interval_recency_decay`). El camino real para FIESTA NO es recalibrar más
fino — es item #1 de la agenda pendiente: señal de drift de nivel YoY con
significación, que detecte el régimen nuevo y lo trate como régimen (no
como bias corregible con data histórica).

## Lección de la validación

Un A/B honesto que sale negativo es tan valioso como uno que sale
positivo: descarta la vía «más fina granularidad de calibración» y
re-enmarca el problema como drift de régimen. Dos hallazgos colaterales
atrapados al ejecutar el A/B y corregidos: (a) el knob inicialmente quedó
anidado al nivel equivocado del YAML (experimento nulo; test de regresión
del árbol de config añadido), (b) las llaves-tupla rompían la
serialización JSON del manifiesto (`_jsonable` normaliza a str).

## Non-goals

- No suavizar gates ni clip: la celda estacional recibe la MISMA disciplina
  de evidencia que la banda.
- No tocar selección de campeón ni blends.
- No usar reales del horizonte para calibrar (el horizonte evaluado sigue
  siendo out-of-sample; la celda se alimenta de folds históricos as-of).
