# Proposal — production-readiness

## Contexto

La revisión de ciencia de datos del 19-oct sobre las corridas reales
(`real-20261016-0714`, `futuro-20261016-0726`) concluyó que la metodología era
de primer nivel, pero el motor no estaba listo para producción por cinco
frentes: monitoreo cosmético, banda mal calibrada, sesgo sistemático a leads
largos, inestabilidad del campeón y ausencia del lazo forecast-vs-real.

## Problema

1. **Drift que dispara siempre**: la referencia "primera mitad del histórico"
   comparaba la era de 1-2 hoteles contra el portafolio completo (22-23σ en
   cada corrida) — fatiga de alertas pura; `reference_metrics={}` dejaba el
   drift de métricas (10.1) sin evaluar; `days_since_last=0` hardcodeado y
   `new_records=len(history)` (todo el histórico) hacían decorativa la
   decisión de re-entrenamiento.
2. **Banda de-mediaada**: el supuesto "el sesgo vive en el punto" era falso
   con bias de fold ≠ 0 — CORIN 67% y FIESTA 68% vs 80% nominal.
3. **Sobre-pronóstico estructural a leads largos**: +23%/+30% de bias en
   60-90+ documentado por el backtest.
4. **Churn del campeón por propiedad**: FIESTA seasonal_naive↔lightgbm↔sarimax
   entre corridas con empates DM por todas partes (ruido, no evidencia).
5. **Sin lazo de producción**: la precisión solo se medía dentro de cada
   corrida; el forecast publicado nunca se puntuaba contra los reales.

## Propuesta

- **P0**: registro de estado entre corridas (`run_registry`, append-only
  atómico), drift YoY por propiedad con PSI calibrado, outliers con baseline
  rodante local, re-entrenamiento con estado real, banda conformal
  (cuantiles completos del residuo), tracking vintage + revisión entre
  corridas con detalle por estancia en CSV.
- **P1**: corrección de bias del punto por (propiedad × banda) con shrinkage,
  clip y **gate de consistencia entre folds**; histéresis del campeón
  (incumbente retenido salvo ventaja material); cold start declarado
  (< 365d provisional).
- **Reporte**: auto-explicado (tablero de veredictos con criterio explícito,
  guías de lectura plegables, glosario, cobertura por propiedad) y
  rediseño profesional (membrete de corrida, KPIs serif tabulares, secciones
  agrupadas por orden de lectura, estilo unificado de gráficos).

## Resultados (validado con datos reales, corridas 19-22 oct)

- Alerta perpetua 22σ **eliminada**; hallazgos YoY reales y accionables
  (MARINA PSI 2.67→4.28 entre ventanas: crecimiento genuino detectado).
- MAPE futuro (backtest honesto) 27.45% → **26.63%**; verificación 22.98% →
  **22.04%**; bias a leads 60-90+ de +23/+30% a −2/−9%.
- **Primer vintage real**: el forecast publicado se puntuó contra los reales
  materializados — WMAPE **9.35%** en 78 estancias, mejor que la expectativa
  del propio motor en las tres bandas de lead; revisión entre corridas
  −0.9% (estable).
- Campeón estable 9 corridas (pickup_curve; empates DM → parsimonia); FIESTA
  cambió a lightgbm SOLO con la data nueva (ventaja material) y quedó estable
  después — la histéresis funciona como se diseñó.
- Reproducibilidad demostrada al dígito entre corridas de días distintos con
  la misma data (revisión entre corridas = 0.0%).
- Suite: **399 passed + 1 skipped**.

## Non-goals

- **Orquestación/scheduler** (excluido por decisión del negocio): el
  registro la habilita cuando se active.
- **SHAP para campeones ML** (explainability §SHAP): la maquinaria existe
  (`explain_with_shap`) pero el runner opera el método sustituto declarado.
  Riesgo conocido y latente: solo manifiesta si un modelo ML gana el
  campeonato global — nunca ha ocurrido (campeón histórico: pickup_curve).
  Cablearlo es tarea menor cuando eso ocurra.
- **Revenue/ADR**: fuera de alcance de esta versión (spec forecasting).
