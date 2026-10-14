# Design — per-property-intervals

## Decisions

### D1 — La fuente de residuos es el forecast OPERATIVO, no los candidatos puros
Los intervalos deben describir el error de lo que se publica: campeón +
empalme del blend + perfil DOW. Calibrar con el campeón puro subestima
el tramo largo de las propiedades con blend (el rollout puro tiene más
error que el operativo) y calibrar con revisiones re-fit del backtest
muestreado da 4-8 obs por bucket — insuficiente. Los folds ya predijeron
todo lo necesario; el ensamble operativo es un re-ensamblado de
predicciones existentes, sin re-entrenar nada.

### D2 — La decisión del blend se aplica uniforme a los folds
El blend se decide al corte T0 (colapso del rollout + elección por
evidencia). Para calibrar intervalos, esa decisión se aplica a las 4
ventanas de fold — incluso si a algún corte de fold el colapso no
hubiera disparado. Aproximación deliberada (validada en el
contrafactual): los intervalos describen la configuración que OPERA, no
un contrafactual por fold.

### D3 — La circularidad se rompe reteniendo predicciones
El blend se decide dentro del runner (necesita las métricas de folds);
los residuos operativos necesitan el blend. En vez de duplicar la
decisión fuera, `real_fold_metrics` RETIENE las predicciones de todos
los candidatos y el runner ensambla los residuos DESPUÉS de decidir el
blend. Cero re-entrenamientos, cero duplicación de lógica.

### D4 — Escalera de fallback explícita
(propiedad, banda) con ≥ `min_residuals` → pool de la propiedad → pool
de la banda (portafolio, = comportamiento actual) → global. Las celdas
chicas (1-2, 90+; SJOSL en folds tempranos) no inventan cuantiles con
n=4: caen al escalón con muestra. Sin `fold_predictions` (corridas de
muestra), se usa la fuente actual completa.

### D5 — El forecast puntual no cambia
Este cambio es de calibración de intervalos EXCLUSIVAMENTE. Si algún
test de métricas se mueve, es un bug. El manifiesto declara la fuente
(`interval_source`) para que el consumidor sepa qué está leyendo.

## Risks / Trade-offs

- [Coberturas por celda ruidosas] — cada celda (propiedad, banda) de la ventana T0 evalúa 8-15 días: una cobertura de 25% o 100% en una celda es ±1-3 días. El criterio de aceptación es la escala de anchuras y la cobertura AGREGADA, no celdas individuales.
- [Residuos de folds con predicciones as-of imperfectas] — los folds entrenan con cortes más viejos; sus errores pueden ser ligeramente mayores que los de T0 (sobre-cobertura, p. ej. VILLAS 100%). Preferible a sub-cobrir: una banda honesta es la que contiene al real.
- [Memoria] — fold_predictions ≈ 15K filas: despreciable.
- [Cambio de firma de real_fold_metrics] — tercer valor de retorno; los callers (run_real.main, tests) se actualizan en el mismo cambio.
