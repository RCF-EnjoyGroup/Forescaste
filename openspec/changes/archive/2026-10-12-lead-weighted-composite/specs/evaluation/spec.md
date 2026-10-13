## MODIFIED Requirements

### Requirement: Integridad del benchmark de candidatos
El sistema SHALL evaluar cada candidato con un protocolo íntegro: features de entrenamiento sin el target de la propia fila, rollout multi-step con features as-of correctas, señal de pickup disponible para los candidatos que la soporten y el mismo techo de capacidad aplicado a todos. El composite de selección SHALL ponderar los leads operativos (1 a 30 días) con mayor peso que los leads de planeación (31 a 89), de modo que un candidato plano no gane por "consistencia" en el rango largo mientras destruye el rango corto.

#### Scenario: Features sin target en entrenamiento
- **WHEN** un candidato ML entrena con features rolling
- **THEN** la ventana de cada feature rolling excluye el valor del target de la fila que se predice
- **THEN** una feature que incluye el target de la propia fila es detectada y excluida del entrenamiento

#### Scenario: Rollout multi-step correcto
- **WHEN** un candidato ML predice varios pasos hacia adelante
- **THEN** cada paso usa las features as-of del paso que se predice
- **THEN** el rollout no degrada a un forecast plano por desalineación de filas

#### Scenario: Señal de pickup en candidatos ML
- **WHEN** se entrenan los candidatos ML
- **THEN** disponen de la señal de pickup as-of (RN ya reservados y su evolución) como feature cuando los datos la proveen

#### Scenario: Comparabilidad de candidatos
- **WHEN** el benchmark compara candidatos clásicos y ML
- **THEN** todos se evalúan contra los mismos reales en las mismas ventanas y con el mismo post-procesamiento aplicado en producción

#### Scenario: Candidato pickup representativo
- **WHEN** un candidato basado en curvas de pickup participa del benchmark
- **THEN** su implementación de evaluación es representativa del modelo que operaría en producción (mismo grano y método), sin lo cual el candidato se retira del tablero

#### Scenario: Composite ponderado por lead
- **WHEN** el composite de selección calcula el score de un candidato
- **THEN** los leads operativos (1-30) pesan al menos el doble por día que los leads de planeación (31-89)
- **THEN** un candidato plano en día-de-semana no gana por consistencia en el rango largo si el rango corto muestra que destruye el patrón operativo de la propiedad

#### Scenario: Sin desagregación de lead disponible
- **WHEN** las métricas de fold no traen wmape_short/wmape_long
- **THEN** el composite cae al WMAPE plano de la ventana completa (compatibilidad con corridas de muestra y tests)
