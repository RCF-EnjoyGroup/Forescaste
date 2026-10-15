# forecasting — delta

## MODIFIED Requirements

### Requirement: El campeón opera
El forecast SHALL ser generado por el modelo ganador del benchmark (global y por propiedad). En modo futuro, la evaluación inyectada (métricas, desgloses) SHALL ser operada por el campeón seleccionado **as-of** al inicio de la ventana de backtest con el mismo pipeline de producción (fit/predict/blend) y las mismas restricciones del forecast publicado (piso de reservas, techo de capacidad). El manifiesto SHALL declarar los modelos que operaron la evaluación (`eval_model_by_property`).

#### Scenario: Métricas futuras miden lo que opera
- **WHEN** se ejecuta una corrida futura y una propiedad opera un modelo distinto al campeón global
- **THEN** las métricas publicadas de esa propiedad provienen de predicciones de ESE modelo
- **THEN** la selección usada para operar la evaluación se calcula con folds que terminan ANTES de la ventana evaluada (sin ventaja retrospectiva)

#### Scenario: Métricas evalúan el forecast publicado
- **WHEN** la evaluación futura produce predicciones
- **THEN** se aplican piso de reservas y techo de capacidad antes de medir (la métrica jamás evalúa un forecast físicamente imposible o inferior a lo ya reservado)

### Requirement: Selección por propiedad con evidencia material
La selección por propiedad SHALL retener al campeón global cuando el ganador local no lo supere por un margen de materialidad configurable (default 5% relativo del score compuesto en esa propiedad); la retención SHALL quedar documentada en `property_fallback`.

#### Scenario: Override dentro del ruido
- **WHEN** el ganador local supera al campeón global por menos del margen
- **THEN** la propiedad opera el campeón global y el fallback documenta la retención con los scores

### Requirement: Curva de pickup con bordes de bins explícitos
La curva de pickup SHALL soportar bordes de bins de lead explícitos (`bin_edges`), densos a leads cortos, con el último borde como catch-all; la configuración SHALL validar que sean estrictamente crecientes.

#### Scenario: Gradiente a leads cortos
- **WHEN** la curva se ajusta con bordes densos (p. ej. [1,3,7,14,21,30,...])
- **THEN** leads operativos adyacentes (1 vs 9) reciben pickups esperados distintos cuando los datos los tengan
- **THEN** los leads por encima del último borde usan la curva del bin final (nunca NaN ni fuera de rango)

### Requirement: Baseline estacional suavizado
`seasonal_naive` SHALL usar la media de las últimas K ocurrencias del mismo día de semana (default 4), no una única observación — el benchmark debe exigir a los candidatos superar un baseline no ruidoso.

#### Scenario: Predicción del baseline
- **WHEN** se predice una fecha futura con histórico suficiente
- **THEN** el forecast es la media de las últimas K ocurrencias del mismo DOW previas a la fecha
