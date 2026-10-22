# monitoring — delta

## ADDED Requirements

### Requirement: Estado entre corridas (registro)
El sistema SHALL persistir un registro append-only entre corridas con: métricas, fuente de métricas, campeón global y por propiedad, cobertura de banda y timestamp de cada corrida; el registro SHALL escribirse atómicamente y un registro ausente o corrupto SHALL jamás romper la corrida.

#### Scenario: Drift de métricas con memoria
- **WHEN** existe una corrida anterior registrada con fuente de métricas comparable
- **THEN** el drift de MAPE/WMAPE/Bias se compara contra esa corrida (no contra un placeholder vacío)

#### Scenario: Re-entrenamiento con estado real
- **WHEN** se decide si re-entrenar
- **THEN** los días desde el último entrenamiento y los registros nuevos se calculan del registro (no hardcodeados)

#### Scenario: Histéresis del campeón por propiedad
- **WHEN** la selección por propiedad de esta corrida encuentra un ganador local que no supera materialmente al incumbente de la corrida anterior
- **THEN** el incumbente se retiene y el motivo queda documentado

#### Scenario: Tracking vintage del forecast publicado
- **WHEN** la corrida anterior publicó un forecast operativo y parte de su horizonte ya tiene reales
- **THEN** ese forecast se puntúa por vintage (métricas, desglose por lead y cobertura de banda)
- **THEN** la revisión entre corridas (delta sobre estancias comunes) se publica en los KPIs
- **THEN** el detalle forecast-vs-real por estancia se publica en `vintage_tracking_detail.csv` con la corrida que lo publicó (`published_run_id`)
- **WHEN** ninguna estancia del forecast anterior se ha materializado
- **THEN** el CSV no se publica (nunca se inventa) y el manifiesto lo documenta

## MODIFIED Requirements

### Requirement: Detección de drift
El sistema SHALL monitorear deterioro de MAPE y WMAPE, drift de Bias, drift de features, drift de la curva de pickup y drift de la distribución de demanda; el monitoreo SHALL ejecutarse como parte de la corrida productiva y sus resultados quedar registrados en el manifiesto.

#### Scenario: Drift de demanda estacionalmente comparable
- **WHEN** se evalúa el drift de la distribución de demanda de una propiedad
- **THEN** la referencia es la misma ventana calendario un año atrás (YoY) por propiedad, medida con PSI
- **THEN** una propiedad con menos de 365 días de histórico queda documentada como no evaluable (sin alerta inventada)
- **THEN** una alerta que dispararía en cada corrida por crecimiento estructural antiguo del portafolio NO se genera

#### Scenario: Outliers con baseline local
- **WHEN** se calcula el z-score de un día de una propiedad
- **THEN** el baseline es una ventana rodante centrada en la propia fecha (no la media de todo el histórico)
- **THEN** el arranque de baja demanda de una propiedad nueva no se clasifica como outlier
