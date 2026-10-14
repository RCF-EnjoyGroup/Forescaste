## Why

La revisión de resultados propiedad×mes (2026-10-14) mostró que el **MAPE diario es una métrica deshonesta para propiedades chicas**: SJOSL (media 20.6 RN/día, reales de 4-12 RN) reporta MAPE 51.1% — pero su nivel es impecable (sesgo −0.1 RN) y el error es pura volatilidad diaria que ningún modelo puede ver a leads largos. Los "peores días" del portafolio son todos reales de 4-19 RN donde ±10 RN de error = "+300%".

La validación del grano semanal sobre la ventana T0 demuestra que el ruido se cancela y la señal queda:

| Propiedad | WMAPE diario | **WMAPE semanal** | MAPE diario | **MAPE semanal** |
|---|---|---|---|---|
| SJOSL | 34.3% | **21.5%** | 51.1% | **24.7%** |
| CORIN | 20.7% | 16.4% | 22.2% | 19.2% |
| FIESTA | 11.3% | 6.0% | 12.0% | 6.0% |
| LIREL | 16.8% | 14.2% | 28.3% | 23.3% |
| MARINA | 15.9% | 14.3% | 15.5% | 14.1% |
| VILLAS | 16.0% | 10.4% | 17.9% | 15.3% |

El grano semanal mejora la lectura en TODAS las propiedades (+1.6 a +12.8pp) porque es el grano real de las decisiones operativas (staffing, compras) y cancela el ruido denominador-chico. El reporte actual solo mide al grano diario, con lo cual Revenue lee "SJOSL 51%" cuando la operación semanal está en 21.5%.

Hallazgos descartados con evidencia en la misma revisión (documentados en el design): el sesgo de nivel de MARINA es un giro de demanda POST-corte (jul 26.5 → oct 34.2) que ningún modelo podía ver — lightgbm sigue siendo el mejor disponible según folds; el sobre-pronóstico entresemana de LIREL no es cancelaciones (booked cae solo en 13% de los stays) y pickup_curve es su mejor modelo por 3pp; el condicionamiento por mes de la curva se probó con A/B (folds + T0) y quedó neutro/con señales cruzadas — no se justifica.

## What Changes

- **Nuevo desglose `metrics_by_property_weekly`**: agrega forecast y reales por (propiedad, semana ISO) y mide MAPE/WMAPE/MAE/RMSE/Bias al grano semanal — la lente honesta para propiedades chicas y el grano operativo para todas.
- **Contexto de lectura para el MAPE**: `metrics_by_property` gana las columnas `mean_actual` y `p10_actual`; la tabla del reporte visual marca el MAPE con advertencia cuando el decil inferior de reales diarios cae por debajo de 10 RN (el artefacto denominador-chico).
- **Cableado**: el breakdown entra al `accuracy_report` como `by_property_weekly` y a la tabla del `report.html` como columna "WMAPE semanal" junto a la diaria.
- Sin cambios a métricas de selección, gate, KPIs del manifiesto ni al forecast.

## Capabilities

### Modified Capabilities
- `evaluation`: MODIFIED — Desgloses de precisión: se agrega el grano semanal por propiedad y el contexto de escala de reales para leer el MAPE honestamente.
- `reporting`: MODIFIED — La sección de accuracy del reporte visual muestra WMAPE diaria y semanal con la advertencia de MAPE para propiedades chicas.

## Impact

- **Código**: `src/evaluation/breakdowns.py`, `src/forecasting/runner.py`, `src/reporting/visual_report.py`, tests.
- **Comportamiento observable**: `forecast_accuracy_report.json` gana `by_property_weekly`; la tabla de accuracy del `report.html` gana la columna semanal y la nota de advertencia; nada más cambia.
- **Riesgo**: nulo para el forecast — es reporting puro sobre los mismos datos.
