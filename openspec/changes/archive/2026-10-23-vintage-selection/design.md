# Design — vintage-selection

## El caso que lo expuso (flujo real del 23-oct)

Data nueva (reales al 22-oct) → el usuario corrió futura (corte 22-oct,
cubre 23-oct+) y después verificación. El vintage de la verificación tomó
"la última futura" — cuyo horizonte entero es futuro → `None`. La añada
graduable era la publicación ANTERIOR (corte 19-oct, cubre 20-oct+): 18
estancias ya materializadas. El feature se apagó exactamente en su primer
día de operación real con datos frescos.

## La corrección

`select_gradable_publication`: itera los registros de modo `futuro` del más
reciente al más antiguo (máx. 5) y gradúa la PRIMERA publicación con
scoring no vacío. La semántica queda:

- **Vintage** = la publicación más reciente cuya añada ya tiene reales
  (lo que el negocio recibió y ya se puede juzgar), con `published_run_id`
  declarado en el reporte y el CSV.
- **Revisión entre corridas** (`forecast_vs_previous`) = contra la última
  publicación, sin cambio: mide el linaje operativo (cuánto se movió el
  número publicado entre corridas consecutivas).

Con el flujo operativo natural (refresh → futura → verificación), ambas
lecturas coexisten: la revisión mide el movimiento del linaje; el vintage
mide la añada contra la realidad.
