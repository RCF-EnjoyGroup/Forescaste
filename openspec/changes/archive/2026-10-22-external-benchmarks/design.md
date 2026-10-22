# Design — external-benchmarks

## Por qué juzgar por horizonte y no con un umbral plano

La misma cifra de error significa cosas opuestas según el horizonte: 16%
a 7 días es un problema de higiene de datos (la industria espera <7% porque
el negocio ya está en los libros); 16% a 90 días está dentro de la banda
resort (18-25%). El umbral plano del panel ("< 15% bien") mezclaba ambos.
El desglose agrupa las bandas de lead del accuracy report en tres
horizontes — corto ≤7d, medio 8-30d, largo 31-90d — y junta sus errores con
**pooling honesto**: Σ|e| = Σ(mae×n) y Σ|actual| = Σ(mae×n/wmape) por banda
→ WMAPE del grupo = Σ|e|/Σ|actual|, ponderado por volumen real, no un
promedio simple de ratios.

## Las bandas y su fuente

Perfil **resort/estacional** (el del portafolio Enjoy) del estándar
operativo compilado por HospitalityOS (2026-09, a partir de Hospitality
Net, Lighthouse y Cloudbeds): 7d <7% · 30d 12-18% con acción >22% · 90d
18-25%. Codificado como corto <7% (ajustar 7-12%, deteriorado >12%), medio
<18% (ajustar 18-22%, deteriorado >22%), largo <25% (ajustar 25-30%,
deteriorado >30%) — **configurable** en `monitoring.benchmark_bands`: el
negicio puede calibrar el perfil sin tocar código.

## Tracking signal: el control que faltaba

`TS = Σ error con signo / media|error|` (señal de rastreo clásica de la
cadena de suministro, umbral |TS| ≤ 4). Dentro de una añada: errores
simétricos → TS≈0; un sesgo del +6% sostenido sobre 78 estancias → TS alta
→ "consistente en una dirección", lo que el MAPE (sin signo) jamás
revela. Se calcula en `score_vintage` desde el mismo merge honesto (sin
datos nuevos), viaja en el manifiesto y en el registro — de modo que el
expediente acumule TS por añada para la futura vista de track record. El
regenerador la retro-inyecta a corridas viejas recomputándola del CSV de
detalle (recomputación exacta, nunca estimada).

## Honestidad

- Los veredictos por horizonte requieren n suficiente (≥ 10 obs en el
  grupo); sin datos la fila se omite, no se inventa.
- La fila del WMAPE de ventana se queda (es el número del KPI) pero su
  criterio ahora declara que mezcla horizontes y apunta al desglose.
- La TS de añadas viejas solo aparece si su CSV de detalle existe en
  disco; si no, la fila se omite.
