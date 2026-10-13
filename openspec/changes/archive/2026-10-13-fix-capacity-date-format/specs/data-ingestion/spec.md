## MODIFIED Requirements

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
