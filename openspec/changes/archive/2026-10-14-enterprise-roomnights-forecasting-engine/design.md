## Context

El workspace se reinició desde cero: no existe código, specs ni configs previas en disco (así lo decidió el usuario), aunque el historial de Git conserva iteraciones anteriores del pipeline que sirven como referencia de dominio (esquemas reales `staging.enjoy_fac_hotel` / `public.enjoy_inventory`, bucket de leads, propiedades canónicas). OpenSpec 1.13.0 quedó inicializado con schema `spec-driven` y el cambio `enterprise-roomnights-forecasting-engine`.

Restricciones que moldean el diseño (ver proposal.md - Why):

- Tres fuentes reales exclusivas: `capacity_daily`, `demand_daily`, `pickup_by_lead`. Sin datos externos (clima, competencia, eventos propietarios) — los feriados solo si existe fuente confiable.
- Target `Final Room Nights` a `Property × Stay Date`; Revenue fuera de esta versión.
- Capacidad como restricción física, pickup como señal fundamental, cero leakage.
- Evaluación exclusivamente temporal; métricas MAPE/WMAPE/MAE/RMSE/Bias con desgloses por lead time, propiedad y estacionalidad.

## Goals / Non-Goals

**Goals:**

- Arquitectura modular en `src/` lista para producción: pipeline re-ejecutable con datos actualizados sin tocar código.
- Champion/challenger con los 7 candidatos, selección out-of-sample (global y por propiedad).
- Backtesting por snapshots (T-90 … T-1) y persistencia de revisiones de forecast.
- Intervalos P10/P50/P90, explicabilidad por fecha/propiedad, dataset Power BI, KPIs ejecutivos.
- Data Quality Report + pruebas de leakage que detienen el pipeline ante fallos.
- Reproducibilidad total (seed, versiones, hiperparámetros, timestamps).

**Non-Goals:**

- Capa económica de Revenue (`RN × ADR`) — queda para una versión futura.
- Integración en vivo con Power BI o con la base de producción (se entrega el dataset; la conexión/carga queda fuera).
- Servicio en tiempo real / API de serving; esta versión corre batch.
- Re-entrenamiento automático en producción (se habilita el mecanismo y su configuración, no la orquestación programada).
- Granularidades más finas que `Property × Stay Date` salvo que las fuentes las soporten de forma consistente.

## Decisions

**D1 — Estructura modular por capas, sin lógica en notebooks.**
`src/{data,validation,features,forecasting,models,evaluation,backtesting,explainability,monitoring,reporting}` con contratos de datos tipados (DataFrames validados por esquema en runtime). Los notebooks quedan como interfaz de análisis/presentación que importan de `src/`.
*Alternativa descartada*: notebook único — las iteraciones previas mostraron que la lógica embebida no es testeable ni reproducible.

**D2 — Inspección de esquema guiada por config, no por nombres duros.**
Un módulo de descubrimiento detecta propiedad/fecha/métrica en cada fuente; `config.yaml` solo declara nombres alternativos conocidos (mapping de propiedades, columnas candidatas). Si la detección falla, el pipeline falla con error claro (spec `data-ingestion`).
*Alternativa*: asumir columnas — violaría el requisito de no asumir nombres.

**D3 — Capa de "as-of snapshot builder" como primitiva central anti-leakage.**
Todas las features se construyen mediante un builder que recibe la fecha de corte y solo emite columnas computadas con datos ≤ corte. Las pruebas de leakage (spec `leakage-prevention`) atacan este contrato: mutaciones de prueba que intentan colar datos futuros deben hacer fallar la corrida.
*Por qué*: es más barato garantizar el invariante en un punto único que auditar features individualmente.

**D4 — Champion/challenger con evaluación por ventanas rolling/expanding y métrica de selección WMAPE + |Bias|.**
La selección considera WMAPE, MAE, estabilidad entre ventanas y penalización de bias; el MAPE solo no decide (spec `evaluation`). Selección por propiedad como capa opcional sobre las mismas ventanas.
*Alternativa descartada*: seleccionar por menor MAPE — históricamente produce campeones con sesgo sistemático.

**D5 — Curva de pickup como modelo baseline explícito (no solo feature).**
`Final = Current Bookings + f(lead, propiedad, calendario)` con `f` aprendido de curvas históricas no lineales (bines/monotone GBM). Cumple el baseline obligatorio y aporta la lectura de negocio "¿cuánto pickup nos falta?".
*Por qué sobre regresión lineal de pickup*: el booking curve real es cóncava y varía por lead/propiedad.

**D6 — Dos fases de resultado: `raw_forecast` (estadístico/ML) y `final_forecast = min(raw, capacity)`.**
Se persisten ambas más `capacity` y la alerta de violación (spec `forecasting`), para no ocultar el comportamiento del modelo.

**D7 — Intervalos: conformal sobre residuos de validación temporal cuando el modelo no produce percentiles nativos.**
LightGBM/XGBoost/CatBoost → conformal; SARIMAX/ETS → bandas nativas; Seasonal Naive → cuantiles empíricos de errores históricos. Si un modelo no permite ningún intervalo, se marca explícitamente en la salida (spec `forecasting`).
*Por qué*: cobertura verificable > bandas teóricas no validadas.

**D8 — Evaluación multi-step: forecasts congelados por snapshot, comparación contra RN posteriormente observados.**
El backtester materializa datasets por cutoff y los cachea; las métricas se calculan sobre la unión de snapshots, con desgloses (lead time, propiedad, estacionalidad) derivados del mismo frame.

**D9 — Salida relacional plana para Power BI.**
Una tabla de forecast (columnas del spec `reporting`) + una tabla de accuracy + una de pickup, en CSV/Parquet. Sin modelo estrella ni semántica proprietary: Power BI consume directo.
*Alternativa*: modelo tabular semántico — más fricción para el equipo, sin beneficio en esta versión.

**D10 — Reproducibilidad por "run manifest".**
Cada corrida escribe un manifiesto JSON (versiones de dataset/modelo, ventanas, hiperparámetros, seed, métricas, timestamp) junto a sus salidas; los directorios de corrida se inmutan.

## Risks / Trade-offs

- [Selección por propiedad sobrefittea con pocas ventanas] → exigir mínimo de ventanas/observaciones por propiedad; si no se alcanza, caer a selección global y marcar la propiedad como "sin selección propia".
- [Conformal con pocos folds históricos da bandas anchas o mal calibradas] → reportar cobertura empírica junto a cada banda; degradar a sin-intervalo explícito antes que publicar bandas no validadas.
- [Curva de pickup no estacional en propiedades nuevas (cold start)] → guard de nivel reciente + declaración explícita de baja confianza (el histórico de booking curves de apertura es back-loaded y no estacionario).
- [Detección de esquema ambigua en fuentes nuevas] → fallar temprano con diagnóstico en lugar de adivinar; la config permite declarar mapping cuando la detección no basta.
- [Reinicio desde cero pierde el pipeline anterior validado] → el historial de Git queda intacto como referencia; este cambio no borra historial, solo no restaura archivos. Riesgo aceptado y decidido por el usuario.
- [Costo computacional del bake-off de 7 modelos × propiedades × ventanas] → benchmark completo solo en re-validación; la corrida de producción corre el campeón ya seleccionado (el manifiesto lo referencia).
- [Drift con pocos puntos de referencia al inicio] → arrancar con umbrales conservadores documentados y re-calibrar tras el primer trimestre de corridas.

## Migration Plan

Reinicio greenfield, no hay despliegue previo que migrar:

1. Esqueleto `src/` + config + manifiesto de corrida + Data Quality Report (frena el pipeline ante críticos).
2. Ingesta con detección de esquema + pruebas de leakage verdes.
3. Features + backtester por snapshots + métricas con desgloses.
4. Bake-off de los 7 candidatos → campeón (y selección por propiedad) → capa de capacidad → intervalos.
5. Explicabilidad, salidas de forecast, dataset Power BI, KPIs y reportes.
6. Monitoring (drift, retraining configurable) y documentación técnica final.

Rollback: al ser greenfield batch, "deshacer" es dejar de ejecutar la corrida; ningún sistema depende de ella hasta que Power BI consuma el dataset.

## Open Questions

- ¿Existe ya una fuente confiable de feriados para Costa Rica que debamos consumir, o el indicador se omite en v1? (No cambia specs: ambos caminos están contemplados en `feature-engineering`.)
- ¿El equipo preiere Parquet o CSV como formato del dataset Power BI? (Ambos cubren el spec; se decide en tasks/implementation.)
