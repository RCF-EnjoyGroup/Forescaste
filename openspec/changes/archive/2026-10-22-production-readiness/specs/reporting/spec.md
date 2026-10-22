# reporting — delta

## ADDED Requirements

### Requirement: Reporte auto-explicado
El reporte HTML SHALL ser legible sin conocimiento previo del motor: cada sección de gráfico SHALL llevar una guía de lectura plegable, un glosario de términos SHALL estar presente, y un panel de interpretación y validación SHALL traducir las métricas de la corrida a veredictos con criterio explícito. Los veredictos son ayuda de lectura: los gates duros viven en los criterios de aceptación del runner y en el gate de calidad de datos.

#### Scenario: Panel de interpretación y validación
- **WHEN** la corrida tiene métricas
- **THEN** el panel muestra WMAPE, bias, cobertura de banda (contra el 80% nominal), convergencia del error, significancia del campeón, cold start, drift YoY, estado del registro, forecast publicado vs reales (vintage) y revisión entre corridas
- **THEN** cada fila declara el criterio con el que se juzga y un veredicto visual
- **WHEN** una métrica no es medible en el modo de corrida (p. ej. cobertura en corrida futura)
- **THEN** la fila se presenta como N/D con el motivo, nunca se omite silenciosamente

#### Scenario: Cobertura en la tabla de accuracy
- **WHEN** la corrida evaluó cobertura de banda por propiedad
- **THEN** la tabla de accuracy incluye la columna de cobertura con código de color contra el 80% nominal y marca a las propiedades cold start (provisionales)

#### Scenario: Identidad visual profesional
- **WHEN** se genera el reporte
- **THEN** lleva membrete de corrida con metadatos (corrida, modo, ventana, campeón, generado), KPIs con dígitos tabulares, secciones agrupadas por orden de lectura, estilo unificado de gráficos y es imprimible

#### Scenario: Regeneración de la vista sin re-correr
- **WHEN** se mejora la presentación del reporte
- **THEN** `scripts/regen_report.py` reconstruye `report.html` de una corrida existente desde sus entregables, sin re-entrenar ni re-correr el pipeline
