# Proposal — external-benchmarks

## Contexto

La revisión del estándar de industria (HospitalityOS, «Hotel Forecast
Accuracy Benchmarking», 2026-09) nos da lo que el motor no tenía: una
**referencia externa citable** para juzgar la precisión y una estadística de
control de sesgo acumulado que la industria usa (señal de rastreo, tracking
signal) que complementa el drift de bias puntual.

## Problema

1. **Los veredictos del panel usan umbrales internos planos** ("< 15% bien ·
   15-25% revisar") sin anclaje externo y sin distinguir horizonte: un 16% a
   7 días y un 16% a 90 días son lecturas opuestas de la misma cifra. La
   industria juzga por horizonte: resort/estacional espera <7% a 7d, 12-18%
   a 30d y 18-25% a 90d (acción >22% a 30d).
2. **El sesgo sistemático acumulado no tiene control**: el drift de bias
   compara corridas y el vintage mide la añada, pero ninguna estadística
   acumula el error con signo contra la desviación media — el patrón "3%
   alto todos los meses" que hundió el ciclo presupuestario 2025 de la
   industria es invisible para el MAPE y visible para la señal de
   rastreo (umbral clásico |TS| ≤ 4).

## Propuesta

- **Bandas de benchmark por horizonte en el panel**: el WMAPE se desglosa
  por horizonte (corto ≤7d, medio 8-30d, largo 31-90d) y cada grupo se
  juzga contra su banda de industria (configurable en
  `monitoring.benchmark_bands`, perfil resort/estacional por defecto),
  con pooling honesto (Σ|e|/Σ|actual| entre bandas del grupo). La fila
  del WMAPE de ventana pasa a declarar que mezcla horizontes.
- **Tracking signal del forecast publicado**: `score_vintage` agrega
  `tracking_signal = Σerror / MAD` a la añada (y al registro); el panel
  la publica con el criterio |TS| ≤ 4 en control. Detecta sesgo
  sistemático dentro de la añada mucho antes que el MAPE.
- El regenerador retro-inyecta la TS a corridas viejas desde su
  `vintage_tracking_detail.csv` (recomputación exacta, sin inventar).

## Non-goals

- La vista de expediente vintage-sobre-vintage (track record acumulado):
  el registro ya acumula las TS por añada; la vista llega cuando existan
  4-6 añadas.
- Traducción a dólares del error (CPOR de Enjoy): requiere datos de
  Finanzas — item separado con el negocio.
- Fuentes externas (clima/eventos/cancelaciones) para leads 30-90:
  item del contrato de datos con Sistemas.
