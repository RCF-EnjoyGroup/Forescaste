# reporting — delta

## MODIFIED Requirements

### Requirement: Reporte auto-explicado
El reporte HTML SHALL ser legible sin conocimiento previo del motor: cada sección de gráfico SHALL llevar una guía de lectura plegable, un glosario de términos SHALL estar presente, y un panel de interpretación y validación SHALL traducir las métricas de la corrida a veredictos con criterio explícito. Los veredictos son ayuda de lectura: los gates duros viven en los criterios de aceptación del runner y en el gate de calidad de datos.

#### Scenario: Panel de interpretación y validación
- **WHEN** la corrida tiene métricas con desglose por banda de lead
- **THEN** la precisión se juzga por horizonte — corto (≤6d), medio (7-29d), largo (30d+) — contra bandas de la INDUSTRIA configurables (perfil resort/estacional por defecto: <7% corto, <18% medio, <25% largo; estándar operativo HospitalityOS 2026), con pooling ponderado por volumen entre las bandas de lead del grupo (Σ|e|/Σ|actual|, nunca promedio simple de ratios); grupos con < 10 observaciones se omiten
- **THEN** el WMAPE de ventana declara que mezcla horizontes y apunta al desglose por grupo
- **WHEN** existe tracking vintage con señal de rastreo por propiedad
- **THEN** el panel publica la fila «Tracking signal por propiedad (añada)» con la peor encabezando, el detalle por hotel en el criterio y el umbral configurado
