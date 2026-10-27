# data-ingestion — delta

## MODIFIED Requirements

### Requirement: Validación de calidad de datos
El sistema SHALL validar antes de entrenar: fechas faltantes, duplicados propiedad/fecha, RN negativos, capacidad negativa, capacidad igual a cero, RN mayores a la capacidad, cambios inesperados de capacidad, periodos históricos ausentes, snapshots duplicados, lead times inválidos o negativos e inconsistencias de pickup. Las filas completamente vacías SHALL reportarse como advertencia de higiene y excluirse de los demás checks sin detener la corrida.

#### Scenario: Continuidad diaria por semántica de fuente
- **WHEN** se evalúan fechas faltantes en una serie física diaria (demanda, capacidad)
- **THEN** los huecos dentro del rango propio de cada propiedad se reportan como advertencia
- **WHEN** se evalúa el pickup (registro disperso de eventos)
- **THEN** una fecha de estancia sin fila NO se reporta como hueco (un stay sin reservas legítimamente no trae fila — exigir continuidad diaria al pickup genera falsos positivos por diseño)
- **THEN** las fechas ilegibles del pickup SÍ se reportan como error (higiene de datos aplica a todas las fuentes)
