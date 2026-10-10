## MODIFIED Requirements

### Requirement: Pruebas automáticas de leakage
El sistema SHALL ejecutar pruebas automáticas que detecten: RN futuros, pickup futuro, capacidad futura, features calculadas con el dataset completo, rolling con observaciones futuras y splits aleatorios; si detecta leakage, el pipeline MUST fallar. La suite completa SHALL ejecutarse dentro de la corrida productiva, con todas las fuentes del modelo dentro del perímetro as-of.

#### Scenario: Leakage detectado
- **WHEN** una prueba de leakage encuentra una feature con información posterior a la fecha de corte
- **THEN** la corrida termina en error indicando la feature y la evidencia
- **THEN** no se publica ningún forecast de esa corrida

#### Scenario: Sin leakage
- **WHEN** todas las pruebas de leakage pasan
- **THEN** el pipeline continúa y registra el resultado de las pruebas en el reporte de la corrida

#### Scenario: Suite completa en la corrida productiva
- **WHEN** se ejecuta la corrida productiva end-to-end
- **THEN** la suite de leakage se ejecuta con las fuentes reales de la corrida (demanda, pickup y capacidad) dentro del perímetro as-of
- **THEN** los checks de features calculadas con dataset completo y de rolling con observaciones futuras corren sobre las features reales de la corrida, no únicamente en la suite de tests

#### Scenario: Capacidad post-corte acotada
- **WHEN** la fuente de capacidad contiene fechas posteriores al corte
- **THEN** la capacidad post-corte se usa exclusivamente como techo del horizonte de pronóstico, nunca como feature de entrenamiento ni dentro de agregados as-of
- **THEN** el uso de capacidad futura como restricción queda declarado en la corrida

#### Scenario: Leakage vía matriz de entrenamiento ML
- **WHEN** un candidato ML construye su matriz de entrenamiento a partir de features
- **THEN** una verificación de leakage del paso features→matriz de entrenamiento excluye columnas que contengan el target de la propia fila
