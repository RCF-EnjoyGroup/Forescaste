# data-ingestion Specification

## Purpose

Inspeciona y valida las tres fuentes reales del motor (`capacity_daily`, `demand_daily`, `pickup_by_lead`) detectando su esquema real sin asumir nombres de columna, y emite un Data Quality Report que detiene el entrenamiento ante problemas críticos.

## Requirements

### Requirement: Inspección automática de esquema
El sistema SHALL leer la estructura de cada fuente y detectar automáticamente tipos, llave de propiedad, fecha, campo de capacidad y campo de Room Nights, sin asumir nombres de columnas. El parseo de fechas SHALL manejar de forma transparente tanto ISO (YYYY-MM-DD) como DD/MM/YYYY, eligiendo por columna el formato que resuelva más fechas, de modo que una fuente en formato DD/MM no pierda fechas ni cruce día y mes.

#### Scenario: Detección de llaves y campos
- **WHEN** se carga `capacity_daily`, `demand_daily` o `pickup_by_lead`
- **THEN** el sistema identifica la columna de propiedad, la columna de fecha y el campo numérico correspondiente (capacidad, RN observados, pickup/RN por lead)
- **THEN** cualquier columna no reconocida se reporta sin romper la carga

#### Scenario: Estructura desconocida
- **WHEN** una fuente no contiene las columnas mínimas necesarias (propiedad, fecha, métrica)
- **THEN** el sistema emite un error de validación que detiene el pipeline antes de generar features o entrenar

#### Scenario: Fuente con fechas DD/MM/YYYY
- **WHEN** una fuente exporta sus fechas en formato DD/MM/YYYY (estándar del negocio)
- **THEN** el motor parsea la columna con el formato que resuelve más fechas y NO descarta fechas por día > 12
- **THEN** las fechas ambiguas (día y mes ≤ 12) se interpretan con día primero, consistente con el resto de la columna
- **THEN** una fuente ISO existente parsea idéntico que antes (sin cambio de comportamiento)

#### Scenario: Columna de fecha parcialmente ilegible
- **WHEN** una propiedad tiene filas con fechas presentes pero ilegibles por cualquier formato
- **THEN** el Data Quality Report lo reporta como error sin que la validación crashee
- **THEN** las propiedades con fechas legibles continúan validándose con normalidad


### Requirement: Validación de calidad de datos
El sistema SHALL validar antes de entrenar: fechas faltantes, duplicados propiedad/fecha, RN negativos, capacidad negativa, capacidad igual a cero, RN mayores a la capacidad, cambios inesperados de capacidad, periodos históricos ausentes, snapshots duplicados, lead times inválidos o negativos e inconsistencias de pickup. Las filas completamente vacías SHALL reportarse como advertencia de higiene y excluirse de los demás checks sin detener la corrida.

#### Scenario: Detección de duplicados y negativos
- **WHEN** existen registros duplicados por (propiedad, fecha) o valores negativos de RN/capacidad
- **THEN** el Data Quality Report registra la cantidad de registros afectados, las propiedades y las fechas involucradas

#### Scenario: RN por encima de la capacidad
- **WHEN** un RN observado supera la capacidad disponible de la misma fecha
- **THEN** el sistema lo clasifica como advertencia o error según su frecuencia y lo incluye en el reporte

#### Scenario: Filas completamente vacías
- **WHEN** la fuente contiene filas con todos sus campos vacíos (artefactos de exportación)
- **THEN** el sistema las reporta como advertencia `empty_rows` con su cantidad
- **THEN** dichas filas no generan duplicados espurios ni detienen la corrida


### Requirement: Data Quality Report
El sistema SHALL generar un Data Quality Report con status, número de registros, número de errores, número de advertencias, propiedades afectadas y fechas afectadas.

#### Scenario: Reporte completo
- **WHEN** termina la fase de validación
- **THEN** el reporte incluye las métricas de calidad por fuente y una lista concreta de errores y advertencias


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


### Requirement: Propiedades cerradas excluidas
El sistema SHALL soportar la exclusión configurable de propiedades cerradas (`data.excluded_properties`); las propiedades excluidas no participan de ingesta, calidad, features, benchmark, forecast, backtest ni KPIs, y la exclusión SHALL quedar registrada en el manifiesto.

#### Scenario: Exclusión aplicada en toda la corrida
- **WHEN** una propiedad está en `excluded_properties`
- **THEN** no aparece en la tabla de forecast, el benchmark, el backtest ni la cobertura de evaluación
- **THEN** el manifiesto declara la lista de propiedades excluidas

#### Scenario: Exclusión tras alias
- **WHEN** una propiedad cerrada aparece bajo un alias distinto en alguna fuente
- **THEN** el alias se resuelve primero y la exclusión se aplica al nombre canónico
- **THEN** ninguna fila de la propiedad excluida cruza al pipeline


### Requirement: Fuente opcional de eventos
El sistema SHALL soportar una fuente opcional de eventos privados (`data.events_path`, columnas `event_name`, `event_date` y `property` opcional); cuando el archivo exista SHALL validarse (fechas legibles, sin duplicados por nombre+fecha+propiedad) y alimentar el calendario compartido, y cuando no exista el feature SHALL omitirse con la ausencia documentada en la corrida. Las fechas de eventos aceptan ISO y DD/MM/YYYY (estándar del negocio, con día primero al desempatar ambigüedades).

#### Scenario: Archivo de eventos presente
- **WHEN** `events.csv` existe y es válido
- **THEN** las fechas de eventos entran al calendario de la corrida (feature de distancia de ML y ventanas de la curva)
- **THEN** el manifiesto declara la fuente activa con su número de eventos

#### Scenario: Archivo de eventos ausente
- **WHEN** `events.csv` no existe
- **THEN** la corrida es idéntica a una sin eventos y el manifiesto declara la fuente ausente

#### Scenario: Archivo inválido
- **WHEN** el archivo contiene fechas ilegibles o duplicados por nombre+fecha+propiedad
- **THEN** la carga falla con un mensaje que identifica el problema ANTES de entrenar

#### Scenario: Fechas de eventos en DD/MM/YYYY
- **WHEN** el archivo trae fechas en DD/MM/YYYY (cargado a mano en el estándar del negocio)
- **THEN** se interpretan con día primero — "05/10/2026" es 5 de octubre, no 10 de mayo
- **THEN** una fecha ilegible para ISO y DD/MM sigue deteniendo la corrida con el número de filas afectadas
