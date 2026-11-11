# Proposal — monthly-report-operative-models

## Contexto (validación del 10-nov)

El Excel de vintages mensuales (`scripts/monthly_vintage_report.py`) se
validó contra la data vigente: 54/54 reales de meses cerrados coinciden
al dígito y los pronósticos se reproducen bit a bit — el archivo está
correctamente creado. Pero la validación destapó dos deudas:

1. **Los modelos van en un diccionario fijo** (`MODELOS`, snapshot de la
   selección operativa del 26-oct). Para los cortes actuales
   (15-ene…15-oct) ese snapshot coincide con la selección vigente de la
   época (MARINA giró el 27-oct, FIESTA el 30-oct — ambos después del
   corte 15-oct), pero **para los cortes nuevos (15-nov, 15-dic) usaría
   modelos que ya no operan** (MARINA catboost cuando hoy opera sarimax;
   FIESTA seasonal_naive cuando hoy opera lightgbm). El Excel promete
   "modelos operativos vigentes" — la promesa caduca en el próximo corte.
2. **La hoja de lectura tiene texto muerto**: dice "reales al 25-oct"
   (ahora los reales van al 5-nov) y no declara con qué data ni con qué
   modelos se generó — un archivo viejo no es auditable solo.

## Propuesta

1. **Selección por corte desde el registro**: para cada corte, usar la
   selección operativa (model_by_property) de la última corrida `futuro`
   con `cutoff <= corte` — la misma regla de "memoria por línea"
   (openspec 2026-11-10-registry-mode-memory) aplicada al anclaje
   histórico. Cortes anteriores al motor (ene→jul): el registro más
   antiguo disponible, declarado en la hoja de lectura. Sin registro
   (archivo ausente): el snapshot histórico como fallback, declarado.
2. **Sello de versión**: hoja de lectura con fecha de generación, fecha
   de data usada, la regla de selección aplicada y el mapa de modelos por
   propiedad (más cuántos mapas distintos hubo entre cortes) — así dos
   archivos de fechas distintas son comparables sin adivinar.
3. **Texto vivo**: la línea de "reales al" se calcula de la data.

## Validación

- Tests: (a) anclaje por corte — un corte nuevo toma la selección vigente,
  un corte viejo toma la de su época; (b) cortes pre-motor caen al
  registro más antiguo con declaración; (c) sin registro, fallback
  documentado; (d) el sello declara data-usada y modelos.
- Regeneración del entregable: octubre cerrado (reales al 5-nov) y
  pronósticos de cortes existentes **bit a bit** (mismos modelos de
  época: la anclaje los reproduce).

## Non-goals

- No cambiar la reconstrucción as-of ni el cálculo de la proyección de
  cierre.
- No tocar el motor ni el registro.
