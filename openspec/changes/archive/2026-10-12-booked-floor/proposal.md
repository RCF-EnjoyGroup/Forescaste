## Why

La verificación de FIESTA (análisis hotel por hotel) expuso dos defectos estructurales del motor: (1) el campeón elegido para FIESTA (sarimax) es un modelo del lado demanda que **ignora la señal de reservas en libros** — pronosticó por debajo de lo ya reservado en 20 de 90 días (hasta −39 RN la víspera de una llegada con 404 booked), indefendible en operaciones; (2) la **selección por propiedad descartó a pickup_curve para FIESTA** por un artefacto de disponibilidad de datos: su historia de pickup empieza el 2025-06-14, así que en los folds tempranos entrenaba con ~24 días y el composite la enterró, aunque en la ventana de verificación le gana a sarimax en TODAS las bandas de lead (WMAPE 0.130 vs 0.141; leads 1–6: 0.014–0.020 vs 0.064–0.071) y jamás cae debajo del booked.

## What Changes

- **Piso de reservas en libros** (capa de restricciones, todos los campeones): `final = min(max(raw, current_booked_rn), capacity)` cuando hay booked; sin booked → sin piso. La capacidad sigue mandando: si booked > capacidad (inconsistencia de datos ya documentada), el techo gana y se emite alerta `booked_over_capacity`. La banda P10/P90 NO se pisa (las cancelaciones permiten que el percentil bajo quede debajo del booked); el piso aplica al punto publicado y su p50.
- **Manifiesto + validación de tabla**: `n_booked_floor_alerts`, `n_booked_over_capacity` y `n_booked_floor_applied` declarados; `validate_forecast_table` verifica la fórmula completa floor+techo con manejo correcto de NaN en booked.

> **Nota de implementación**: la exclusión de cold-start del benchmark y el
> umbral `MIN_CURVE_STAYS` se revirtieron después de la corrida real de
> verificación: la interacción entre ambos cambios el campeón de VILLAS a
> catboost (80.7% MAPE, desde 18%) — un efecto dominó que degrada el
> portafolio entero. El piso solo ya resuelve el problema operativo de
> FIESTA (forecast bajo booked); el problema de selección (FIESTA elige
> sarimax sobre la mejor pickup_curve por historia corta) queda documentado
> como deuda separada para un cambio propio con validación de que ninguna
> propiedad regresa.

## Capabilities

### New Capabilities

<!-- Ninguna. -->

### Modified Capabilities

- `forecasting`: MODIFIED — Restricción de capacidad física: la restricción operacional es ahora piso de booked + techo de capacidad, con alertas separadas y la inconsistencia booked>capacidad reportada.
- `evaluation`: MODIFIED — Integridad del benchmark de candidatos: los fits cold-start no se puntúan; la fila de portafolio exige viabilidad en todas las propiedades evaluadas.
- `feature-engineering`: MODIFIED — Curvas de pickup no lineales: la curva en frío (< 60 estancias) degrada a nivel medio documentado.

## Impact

- **Código**: `src/forecasting/constraints.py` (piso en la capa de restricciones), `src/forecasting/core.py` (`MIN_CURVE_STAYS` en `_fit_pickup`), `src/forecasting/run_real.py` (skip de cold-start + regla de portafolio), `src/reporting/forecast_table.py` (fórmula de validación), `src/forecasting/runner.py` (alertas + manifiesto + doc técnica), tests.
- **Comportamiento observable**: ningún forecast publicado queda por debajo del booked (cuando existe); la selección de FIESTA puede cambiar de campeón; los números publicados cambian (mejora esperada en FIESTA: leads 1–6 y días con booked alto).
- **Riesgo**: el piso no modela cancelaciones (conservador en fechas con cancelaciones de última hora — medido en FIESTA: booked final vs real difiere >10 RN en 34/480 estancias); documentado como trade-off, la alternativa (piso por tasa de cancelación) requiere un modelo de cancelaciones que no existe.
