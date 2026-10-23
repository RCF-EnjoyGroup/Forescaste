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

## CORRECCIÓN (23-oct): el A/B estaba confundido y el estimador tenía un side effect

La revisión de las corridas del 23-oct (misma data, métricas idénticas al
dígito) expuso que las coberturas diferían del baseline sin que el punto
cambiara: la maquinaria de recencia cambió el estimador de cuantiles
(interpolación por punto medio ponderado) también con pesos UNIFORMES,
moviendo coberturas por sí sola (LIREL 93.3→95.6%, CORIN 82.2→83.3%,
desviación media 13.15→13.70pp). Eso contaminó el A/B: el "antes"
(`real-20261021-0901`) usaba el estimador viejo y el "después" el nuevo.

**Comparación limpia** (mismo estimador ambos lados):
- estimador nuevo + uniforme (23-oct): desviación media 13.70pp
- estimador nuevo + decay 0.7 (22-oct): desviación media 13.52pp

→ El decay es un **empate técnico** (−0.2pp), no la degradación que se
registró. La decisión no cambia — default uniforme por parsimonia, sin
beneficio demostrado — pero el registro queda corregido. 

**Reparación**: `_wquantile` cae a `np.quantile` exacto con pesos uniformes
— el knob de recencia es ahora puramente aditivo (decay=1.0 reproduce bit
a bit la banda clásica), con test de regresión que lo blinda.

## La lección (va con la corrección)

El A/B midió con datos — y la medición misma tenía un confounder que solo
la corrida de control del día siguiente expuso. Es exactamente por esto
que la reproducibilidad bit a bit es un gate de aceptación: el punto no
cambió, las bandas sí, y eso delató el side effect. Registrar la
corrección evita que el expediente mienta. Los veredictos que se quedan
(TS de FIESTA/SJOSL, coberturas de LIREL/MARINA) siguen siendo señales
reales — y ahora se miden con el estimador correcto.
