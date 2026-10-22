# forecasting — delta

## MODIFIED Requirements

### Requirement: Intervalos de incertidumbre
El sistema SHALL generar P10, P50 y P90 (o lower/forecast/upper) siempre que el modelo lo permita, siendo el P50 el forecast principal; la banda SHALL anclarse al forecast publicado, calibrarse por (propiedad, banda de lead) con los residuos del forecast operativo y respetar el techo de capacidad. La amplitud de la banda SHALL reflejar la escala de error de cada propiedad — no la del portafolio agregado.

#### Scenario: Calibración conformal (cuantiles completos)
- **WHEN** los residuos de una celda tienen sesgo o asimetría (mediana ≠ 0)
- **THEN** los desplazamientos de la banda son los cuantiles completos del residuo (−q90, −q10): la banda contiene el 80% nominal de la muestra de calibración por construcción, sesgo incluido
- **THEN** el sesgo sistemático se corrige en el PUNTO (corrección de bias por propiedad × banda), no ignorándolo en la banda

#### Scenario: Contrato monótono ante sesgo extremo
- **WHEN** más del 90% de los residuos de una celda comparten signo y la banda honesta quedaría a un lado del punto
- **THEN** la banda se proyecta al contrato p10 <= p50 <= p90 y la proyección queda documentada

## ADDED Requirements

### Requirement: Corrección de bias del punto por (propiedad × banda)
El punto publicado SHALL corregirse multiplicativamente por celda (propiedad × banda de lead) cuando los folds del forecast operativo documenten sesgo sistemático ESTABLE: mismo signo en ≥ 75% de los folds (gate de consistencia), shrinkage hacia 1.0 por tamaño de muestra y clip documentado; la corrección se aplica ANTES del piso de reservas y el techo de capacidad; celdas sin evidencia suficiente y correcciones inmateriales no se tocan; el factor viaja en el dataset publicado (`bias_correction_factor`) y el detalle por celda en el manifiesto.

#### Scenario: Sobre-pronóstico estable a leads largos
- **WHEN** una celda documenta sobre-pronóstico consistente entre folds
- **THEN** el punto se corrige a la baja con el factor shrinked y la banda se recalibra con los residuos CORREGIDOS (la banda mide el pipeline completo que se publica)

#### Scenario: Sesgo inestable entre folds
- **WHEN** el signo del bias de una celda gira entre folds (consistencia < 75%)
- **THEN** la celda no se corrige y el motivo queda documentado (sesgo que gira es régimen, no evidencia)
