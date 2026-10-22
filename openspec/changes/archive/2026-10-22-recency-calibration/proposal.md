# Proposal — recency-calibration

## Contexto

Los veredictos no-verdes del panel se triagiaron en tres familias: señales
reales que la próxima añada re-medirá (TS de FIESTA/SJOSL — la añada del
6-oct erró unidireccional), lecturas operativas que se quedan (drift YoY de
MARINA, decisión del negocio) y **dos familias arreglables con ingeniería**:
la calibración de banda anclada a folds de hasta un año de antigüedad, y la
presentación que no cita la medición más fresca disponible.

## Problema

1. **Banda anclada al año pasado**: los cuantiles conformales ponderan igual
   un residuo del fold de hace 11 meses que uno del trimestre reciente. En
   un portafolio en transición de régimen (recuperación post-2025), la
   banda queda anclada al régimen viejo: LIREL 93% y VILLAS 92%
   sobre-cubren (banda ancha heredada), FIESTA 52% y MARINA 63% sub-cubren.
2. **El veredicto corto no cita la medición fresca**: el corto del modo
   futuro usa el backtest as-of (12% en 3-6d), pero la añada PUBLICADA
   midió 5.5% real en esos mismos leads — el lector ve "Ajustar" sin
   saber que la medición más reciente está en banda.
3. **La fila TS no distingue hoteles provisionales**: SJOSL fuera de
   control es su sesgo cold-start documentado; sin la marca, se lee igual
   que un hotel estable.

## Propuesta

- **Cuantiles ponderados por recencia** en la banda: los residuos del
  forecast operativo llevan peso `decay^(edad del fold)` (default 0.7,
  configurable `monitoring.interval_recency_decay`; 1.0 = comportamiento
  anterior). Contrato extendido: las celdas de `residual_bands` pueden
  llegar como DataFrame (residual, weight) — Series sigue válido
  (back-compat con el pool del backtest y tests).
- **Panel**: la fila TS marca hoteles provisionales (⚠); las filas de
  benchmark por horizonte citan la añada publicada reciente cuando cubre
  leads del grupo ("añada reciente: 5.5%").

## Non-goals

- Los umbrales de TS/bandas NO se tocan: los veredictos son medición, no
  opinión. FIESTA/SJOSL re-gravan con la próxima añada (operador y
  correcciones ya cambiaron).
- Ponderación de recencia en la SELECCIÓN de campeón (folds del
  benchmark): cambio mayor, requiere su propio A/B multi-corrida.
- Drift YoY de MARINA: lectura operativa por decisión del negocio.
