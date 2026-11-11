# Proposal — registry-mode-memory

## Contexto (dos evidencias)

1. **Contaminación observada en un A/B real**: el 5-nov, el brazo B del
   A/B de la capa de bias (`real-20261105-1442`) heredó su incumbente de
   `futuro-20261104-1525` — una corrida FUTURA ejecutada después de la
   real del 4-nov. MARINA publicó `sarimax` mientras el baseline publicaba
   `lightgbm`: el A/B quedó confundido (modelo distinto entre brazos) y
   la mejora de MARINA no pudo atribuirse a la capa. El motivo:
   `previous_run = latest_record(registry)` toma el último registro
   agregado SIN filtrar modo, y las dos líneas (verificación real /
   publicación operativa futura) comparten el archivo de registro.
2. **Revisión externa** (Claude): "Filtrar previous_run por cutoff < corte
   actual, no por el último agregado… `select_gradable_publication`
   (vintage) ya filtra modo='futuro'; la histéresis y la memoria de
   monitoreo no".

## Diagnóstico de consumidores

| Consumidor | Qué necesita la memoria | Riesgo actual |
|---|---|---|
| Histéresis del campeón (`incumbent`) | el último estado de SU PROPIA línea (real→real, futuro→futuro) | una futura reemplaza el incumbente de la real (y viceversa): modelos arrastrados entre líneas |
| Memoria de monitoreo (drift de métricas) | la corrida anterior COMPARABLE (misma ventana, misma métrica) | la real compara sus métricas de horizonte contra el backtest inyectado de una futura: ventanas distintas |
| Estado de re-entrenamiento | días desde la última corrida de su línea | una futura manual cuenta como re-entrenamiento de la línea de verificación |
| `previous_operational` + tracking vintage | la última futura | YA correcto (filtra `mode="futuro"`) |

## Propuesta

1. `latest_record(registry, mode=<modo de la corrida>)` como
   `previous_run` de los tres consumidores con memoria — sin fallback
   cruzado: si no hay registro del mismo modo, la memoria arranca vacía
   (documentado en el manifiesto), jamás hereda de la otra línea.
2. El replay del registro real con la regla nueva documenta el caso
   1105-1442: old → `futuro-20261104-1525` (sarimax, contaminante),
   new → `real-20261104-1514` (lightgbm, comparable) — el A/B habría
   quedado limpio.
3. Los consumidores que ya aíslan modo (`previous_operational`,
   `select_gradable_publication`) quedan como referencia del patrón.

## Validación

- Tests: (a) con registro mixto, la corrida real toma la última real
  aunque la futura sea más reciente; (b) sin registro de su modo →
  memoria vacía, sin contaminación; (c) la línea futura sigue intacta.
- Suite completa en verde.
- Replay sobre el registro real: tabla de las últimas corridas con
  previous_run old-vs-new.

## Non-goals

- NoCambiar el registro (append-order, atómico) ni `previous_operational`.
- No tocar la selección de campeón ni los gates de histéresis existentes.
