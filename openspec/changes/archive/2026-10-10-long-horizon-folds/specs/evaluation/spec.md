## MODIFIED Requirements

### Requirement: Validación temporal exclusiva
El sistema SHALL usar exclusivamente validación temporal (expanding window / rolling origin); el split aleatorio train/test está prohibido. Los folds de selección SHALL cubren tramos distintos del año, sin solapar la ventana de métricas publicada.

#### Scenario: Ventanas expansivas
- **WHEN** se configuran las ventanas de validación
- **THEN** cada ventana entrena solo con datos anteriores a ella
- **THEN** la validación simula el comportamiento real de producción

#### Scenario: Cobertura estacional de los folds
- **WHEN** el histórico abarca múltiples ciclos estacionales
- **THEN** los folds de selección cubren el horizonte operativo completo (leads 1 a `fold_horizon_days`) y tesian el año anterior al corte, de modo que cada estación queda evaluada
- **THEN** el campeón no se decide con folds concentrados en una sola temporada ni en una sola banda de lead

#### Scenario: Separación selección/reporte
- **WHEN** se publican métricas de una corrida de verificación
- **THEN** la ventana usada para seleccionar el campeón no solapa la ventana de métricas publicada
- **THEN** ambos períodos quedan declarados en el manifiesto

### Requirement: Significancia estadística de la selección
El sistema SHALL reportar si la ventaja del campeón sobre el subcampeón es estadísticamente distinguible del ruido, con la corrección de autocorrelación alineada al largo real de la ventana de evaluación.

#### Scenario: Ventaja no significativa
- **WHEN** la diferencia de error entre el campeón y el subcampeón no supera una prueba de significancia sobre los folds
- **THEN** el reporte lo declara como empate estadístico
- **THEN** el desempate usa criterios declarados (parsimonia, interpretabilidad, estabilidad) y queda documentado

#### Scenario: Ventaja significativa
- **WHEN** la diferencia de error supera la prueba de significancia
- **THEN** el reporte declara la ventaja como significativa con la estadística y el valor-p

#### Scenario: Corrección HAC al largo real de la ventana
- **WHEN** las ventanas de fold cubren ``fold_horizon_days`` pasos hacia adelante
- **THEN** la corrección de autocorrelación de la prueba usa ese largo (no un valor fijo menor), de modo que los errores multi-paso correlacionados no inflen la significancia
