# Design — recency-calibration

## La hipótesis

En un portafolio en transición de régimen (recuperación post-2025), el
trimestre reciente describe mejor el error vigente que el promedio del año
previo: ponderar los residuos de la banda por `decay^(edad del fold)`
debería re-centrar los cuantiles conformales en el régimen actual y corregir
las coberturas desviadas (LIREL/VILLAS sobre-cubriendo con banda heredada
ancha; FIESTA/MARINA sub-cubriendo).

## La mecánica

- Celdas del forecast operativo salen como DataFrame `(residual, weight)` —
  Series sin peso sigue válida (back-compat con el pool del backtest y con
  tests directos de `build_intervals`).
- `_wquantile`: interpolación sobre pesos acumulados (`cumsum − 0.5·w`,
  normalizado) — con toda la masa en el fold reciente, la banda se pega a
  ese régimen; con pesos uniformes, es exactamente el cuantile clásico.
- La escalera de pools preserva los pesos al concatenar (una celda sin
  peso se une con pesos 1.0).
- El contrato p10 ≤ p50 ≤ p90 sigue proyectándose y documentándose cuando
  la banda ponderada queda unidireccional (el test lo blinda).

## El A/B y el resultado negativo

Corrida `real-20261022-1506` con decay 0.7, misma data/corte/seed que el
baseline `real-20261021-0901` (WMAPE idéntico al dígito — solo bandas
cambian, punto intacto):

| Propiedad | Antes | Después | |
|---|---|---|---|
| FIESTA | 52% | 57% | mejora (la teoría funcionó) |
| VILLAS | 92% | 90% | mejora |
| CORIN | 82% | 84% | leve empeora (sigue OK) |
| SJOSL | 73% | 73% | igual (hereda pools) |
| LIREL | 93% | 97% | empeora |
| MARINA | 63% | 60% | empeora |

Desviación media |cov−80%|: **0.131 → 0.135**. La regla pre-registrada
("se conserva si mejora la desviación media sin degradar ninguna propiedad
materialmente") **no se cumple** → el default queda en 1.0. La maquinaria
y el knob quedan: cuando el expediente de vintages acumule evidencia por
añada, recalibrar es cambiar un número en config.yaml, no un proyecto.

## Por qué un resultado negativo es un buen resultado

El A/B mide con datos lo que la intuición sugería — y la refuta en neto.
Registrar el resultado negativo (en tasks.md y en el comentario del config)
evita que alguien reintente la misma hipótesis dentro de un año sin saber
que ya se probó. Los rojos que se quedaron (TS de FIESTA/SJOSL,
coberturas de LIREL/MARINA) son ahora **señales con evidencia en contra de
la corrección barata** — lo que queda es exactamente lo que debe quedar:
esperar la próxima añada, acumular expediente, y atacar con datos frescos
lo que los datos frescos digan.
