## Purpose

Define cómo el motor explica cada forecast con drivers comprensibles para Revenue Management y gerencia, usando SHAP, importancia de features y contribuciones cuando aplique.

## ADDED Requirements

### Requirement: Explicación por fecha y propiedad
El sistema SHALL explicar el forecast de cada fecha/propiedad mostrando las variables con mayor contribución al resultado.

#### Scenario: Drivers del forecast
- **WHEN** se consulta la explicación de un forecast
- **THEN** se muestran los drivers principales con su contribución (por ejemplo: reservas actuales, pickup histórico, patrón de mismo día de semana, estacionalidad, restricción de capacidad)
- **THEN** la suma de contribuciones es consistente con el forecast reportado

#### Scenario: Contribución de la restricción de capacidad
- **WHEN** el cap de capacidad redujo el forecast
- **THEN** la restricción aparece como contribución negativa explícita en la explicación

### Requirement: Explainability para modelos ML
El sistema SHALL generar explicaciones para los modelos ML mediante SHAP, importancia de features y partial dependence cuando sea útil.

#### Scenario: Corrida SHAP
- **WHEN** el campeón es un modelo ML con soporte de explicación
- **THEN** se producen explicaciones por predicción y agregadas (importancia global)
- **THEN** las explicaciones se guardan como parte de la corrida

### Requirement: Legibilidad para el negocio
El sistema SHALL presentar las explicaciones en lenguaje comprensible para Revenue Managers y gerencia, no solo en términos técnicos.

#### Scenario: Reporte ejecutivo de drivers
- **WHEN** se publica un forecast
- **THEN** la explicación incluye una versión en términos de negocio (reservas, pickup, estacionalidad, capacidad)
- **THEN** los términos técnicos no son el único canal de interpretación
