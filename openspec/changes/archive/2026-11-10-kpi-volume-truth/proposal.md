# Proposal — kpi-volume-truth

## Contexto

El KPI headline de accuracy del reporte publicado mide MAPE global sobre
todas las propiedades mezcladas. En `real-20261106-1323` (verificado
leyendo el `forecast_accuracy_report.json` publicado):

- MAPE global = **34.20%** con WMAPE (ponderado por volumen) = 13.69%;
- MARINA MAPE 58.9% con media real de **6.85 RN/día** (p10=3.0) y SJOSL
  MAPE 66.8% con 18.0 RN/día — ambas = **62.85% del MAPE global sobre
  4.7% de las room nights**;
- SJOSL está declarada **cold start** en el propio manifiesto
  (213d < 365d de histórico: su forecast es provisional por diseño);
- excluyendo esas dos propiedades, el MAPE del resto es **19.95%**.

La tarjeta que lee el negocio muestra 34.2% (y el veredicto contra las
bandas de industria se lee contra ese número) cuando el 95% del volumen
está en ~20%. No es un número falso — es un número que mide otra cosa
(mix de propiedades, no precisión del portafolio).

## Propuesta

Nada se cambia del cálculo del motor: se **publica el contexto junto al
número** (regla del proyecto: nada silencioso, todo etiquetado).

1. `accuracy_report`: bloque `global_excl_cold_start` (métricas
   recalculadas sin las propiedades declaradas cold-start, con la lista y
   la razón de cada exclusión) junto al global — el global NO se toca.
2. KPIs nuevos etiquetados: `forecast_accuracy_mape_excl_cold_start`,
   `forecast_accuracy_wmape_excl_cold_start`,
   `forecast_bias_excl_cold_start`, `n_cold_start_excluded`.
3. Reporte visual: la nota de auditoría de métricas gana una línea
   «excluyendo cold-start (…): MAPE X% / WMAPE Y%» cuando aplique.

## Non-goals

- No cambiar el criterio de aceptación ni las bandas de la config.
- No excluir propiedades del cálculo publicado (sería ocultar, no
  aclarar): se AGREGA el contexto, el global sigue siendo el global.

## Validación

- Test: corrida con una propiedad cold-start → `global_excl_cold_start`
  presente, distinto del global, y con `n_cold_start_excluded` correcto;
  sin cold-start → el bloque no aparece (nada inventado).
- Suite completa en verde.
