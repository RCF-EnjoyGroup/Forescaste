## Why

El rollout recursivo de los GBM (lightgbm/xgboost/catboost) degrada en leads largos: las predicciones se realimentan como lags de los pasos siguientes, y para FIESTA la cascada acumulada arrastra los martes-miércoles de septiembre a ~200-230 RN contra reales de 340-394 (errores de −32% a −43%). Pickup_curve, que calcula cada fecha independientemente (`booked + pickup restante`), no sufre degradación: MAPE 14.6% en leads 31-89 vs 18.8% de lightgbm. En leads 1-30, el GBM es levemente mejor (11.5% vs 11.8%) porque captura el patrón de día-de-semana que la curva agrupa por temporada.

## What Changes

- **Blend de GBM con pickup_curve en `predict_candidate`**: cuando el campeón es un GBM, las fechas con lead > `BLEND_LEAD_SPLIT` (default 30) usan la predicción de pickup_curve en lugar de la del GBM. El GBM conserva los leads 1-30 (donde su rollout tiene lags reales y captura el DOW); la curva opera los leads 31-89 (donde no colapsa).
- **Fitting doble en `build_forecasts`**: cuando el campeón de una propiedad es GBM, también se ajusta pickup_curve para esa propiedad (mismo corte, mismos parámetros del config). El blending es por fila: lead ≤ 30 → GBM; lead > 30 → curva.
- **Manifiesto**: `hyperparameters.per_property_blend` declara qué propiedades operan con blend y el split.
- **Backward compatible**: propiedades cuyo campeón no es GBM no cambian en absoluto.

## Capabilities

### Modified Capabilities

- `forecasting`: MODIFIED — Champion/challenger: cuando el campeón es un GBM, el forecast es un blend que usa el GBM en los leads cortos (1-30) y la curva de pickup en los largos (31+), porque el rollout recursivo del GBM degrada al realimentar sus propias predicciones como lags.

## Impact

- **Código**: `src/forecasting/core.py` (`predict_candidate` blend + `build_forecasts` fitting doble), manifiesto en `src/forecasting/runner.py`, tests.
- **Comportamiento observable**: FIESTA deja de mostrar caídas de ~200 RN en sept-oct; las fechas con lead > 30 usan la señal booked + pickup (inmune a la cascada). Otras propiedades con campeón GBM (MARINA) también se protegen.
- **Riesgo**: pickup_curve puede sobre-estimar fechas de fin de semana a largo lead (vi +17% en sáb de sept) — el techo de capacidad y el piso de booked la contienen.
