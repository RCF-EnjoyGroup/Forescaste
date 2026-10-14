## MODIFIED Requirements

### Requirement: Intervalos de incertidumbre
El sistema SHALL generar P10, P50 y P90 (o lower/forecast/upper) siempre que el modelo lo permita, siendo el P50 el forecast principal; la banda SHALL anclarse al forecast publicado, calibrarse por (propiedad, banda de lead) con los residuos del forecast operativo y respetar el techo de capacidad. La amplitud de la banda SHALL reflejar la escala de error de cada propiedad — no la del portafolio agregado.

#### Scenario: Publicación de percentiles
- **WHEN** el modelo soporta intervalos
- **THEN** la salida incluye p10, p50 y p90 por propiedad y fecha
- **THEN** el p10 <= p50 <= p90 en todos los registros

#### Scenario: P50 coherente con el forecast publicado
- **WHEN** el forecast final fue restringido por capacidad o transformado después de la predicción
- **THEN** el p50 coincide con el forecast publicado en todos los registros
- **THEN** la banda queda anclada al punto publicado, no a la predicción previa a la restricción

#### Scenario: Banda respeta la capacidad
- **WHEN** el p90 excede la capacidad disponible de una fecha
- **THEN** la banda se restringe a la capacidad y la restricción queda registrada

#### Scenario: Calibración por lead
- **WHEN** se publican bandas para un rango de anticipaciones
- **THEN** la amplitud de la banda refleja el error empírico por (propiedad, banda de lead): una propiedad pequeña no hereda la escala de error del portafolio ni viceversa (la banda de un lead 60-89 tampoco hereda la escala de leads 1-30)
- **THEN** los residuos usados son los del forecast operativo (campeón con su blend y su patrón), no los de los candidatos puros
- **THEN** una celda (propiedad, banda) con menos residuos que el mínimo cae a la escalera documentada: pool de la propiedad, luego pool de la banda, luego global

#### Scenario: Origen de los residuos declarado
- **WHEN** se publican bandas
- **THEN** el origen de los residuos usados se reporta de forma veraz (residuos del forecast operativo en folds, revisiones del backtest o fuente sintética etiquetada como tal)

#### Scenario: Modelo sin intervalos nativos
- **WHEN** un candidato no puede producir percentiles
- **THEN** el sistema lo indica en la salida y documenta el método alternativo usado o la ausencia de intervalos
