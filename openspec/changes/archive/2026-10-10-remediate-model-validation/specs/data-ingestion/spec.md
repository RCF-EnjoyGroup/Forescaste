## MODIFIED Requirements

### Requirement: Detención ante problemas críticos
El sistema SHALL detener el entrenamiento cuando exista un problema crítico de calidad de datos, en lugar de producir un forecast potencialmente incorrecto; el gate SHALL aplicarse en la ruta productiva end-to-end.

#### Scenario: Error crítico detectado
- **WHEN** la validación detecta un problema crítico (por ejemplo, fuente vacía, propiedad sin fechas o pickup estructuralmente inválido)
- **THEN** el pipeline se detiene con un mensaje que identifica el problema
- **THEN** no se escribe ningún forecast ni se entrenan modelos en esa corrida

#### Scenario: Sin problemas críticos
- **WHEN** la validación termina solo con advertencias
- **THEN** el pipeline continúa y las advertencias quedan registradas en el reporte

#### Scenario: Gate aplicado en la corrida productiva
- **WHEN** la corrida productiva end-to-end evalúa la calidad de los datos
- **THEN** un status `fail` detiene la corrida antes de entrenar y no publica ningún entregable de forecast
- **THEN** el status de calidad registrado en el manifiesto proviene del mismo gate que decide la detención
