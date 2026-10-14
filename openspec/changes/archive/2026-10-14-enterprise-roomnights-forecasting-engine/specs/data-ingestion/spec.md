## Purpose

Inspeciona y valida las tres fuentes reales del motor (`capacity_daily`, `demand_daily`, `pickup_by_lead`) detectando su esquema real sin asumir nombres de columna, y emite un Data Quality Report que detiene el entrenamiento ante problemas críticos.

## ADDED Requirements

### Requirement: Inspección automática de esquema
El sistema SHALL leer la estructura de cada fuente y detectar automáticamente tipos, llave de propiedad, fecha, campo de capacidad y campo de Room Nights, sin asumir nombres de columnas.

#### Scenario: Detección de llaves y campos
- **WHEN** se carga `capacity_daily`, `demand_daily` o `pickup_by_lead`
- **THEN** el sistema identifica la columna de propiedad, la columna de fecha y el campo numérico correspondiente (capacidad, RN observados, pickup/RN por lead)
- **THEN** cualquier columna no reconocida se reporta sin romper la carga

#### Scenario: Estructura desconocida
- **WHEN** una fuente no contiene las columnas mínimas necesarias (propiedad, fecha, métrica)
- **THEN** el sistema emite un error de validación que detiene el pipeline antes de generar features o entrenar

### Requirement: Validación de calidad de datos
El sistema SHALL validar antes de entrenar: fechas faltantes, duplicados propiedad/fecha, RN negativos, capacidad negativa, capacidad igual a cero, RN mayores a la capacidad, cambios inesperados de capacidad, periodos históricos ausentes, snapshots duplicados, lead times inválidos o negativos e inconsistencias de pickup.

#### Scenario: Detección de duplicados y negativos
- **WHEN** existen registros duplicados por (propiedad, fecha) o valores negativos de RN/capacidad
- **THEN** el Data Quality Report registra la cantidad de registros afectados, las propiedades y las fechas involucradas

#### Scenario: RN por encima de la capacidad
- **WHEN** un RN observado supera la capacidad disponible de la misma fecha
- **THEN** el sistema lo clasifica como advertencia o error según su frecuencia y lo incluye en el reporte

### Requirement: Data Quality Report
El sistema SHALL generar un Data Quality Report con status, número de registros, número de errores, número de advertencias, propiedades afectadas y fechas afectadas.

#### Scenario: Reporte completo
- **WHEN** termina la fase de validación
- **THEN** el reporte incluye las métricas de calidad por fuente y una lista concreta de errores y advertencias

### Requirement: Detención ante problemas críticos
El sistema SHALL detener el entrenamiento cuando exista un problema crítico de calidad de datos, en lugar de producir un forecast potencialmente incorrecto.

#### Scenario: Error crítico detectado
- **WHEN** la validación detecta un problema crítico (por ejemplo, fuente vacía, propiedad sin fechas o pickup estructuralmente inválido)
- **THEN** el pipeline se detiene con un mensaje que identifica el problema
- **THEN** no se escribe ningún forecast ni se entrenan modelos en esa corrida

#### Scenario: Sin problemas críticos
- **WHEN** la validación termina solo con advertencias
- **THEN** el pipeline continúa y las advertencias quedan registradas en el reporte
