# monitoring — delta

## MODIFIED Requirements

### Requirement: Estado entre corridas (registro)

#### Scenario: Memoria por línea de corrida (modo)
- **WHEN** una corrida pide su estado previo al registro (`previous_run`)
- **THEN** lo toma del último registro del MISMO modo (`real` para verificación, `futuro` para publicación operativa, `ad hoc` para corridas sueltas)
- **THEN** la histéresis del campeón, la memoria de drift de métricas y el estado de re-entrenamiento consumen ese registro de su propia línea — jamás el de la otra
- **THEN** si no existe registro previo de ese modo, la memoria arranca vacía y el manifiesto lo declara — nunca herencia cruzada silenciosa entre líneas
- **THEN** los consumidores de la línea operativa (`previous_operational`, `select_gradable_publication` — que ya filtraban `mode="futuro"`) quedan como referencia del patrón, sin cambio de comportamiento
