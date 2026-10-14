# Design — accuracy-aware-blend

## Decisions

### D1 — La evidencia son los folds, no la distribución de predicciones
El trigger de colapso (q25 largo < 0.85 × q25 corto) detecta un problema
de DISTRIBUCIÓN del rollout (los valles de día-de-semana se aplanan).
Ese diagnóstico se conserva intacto. Lo que cambia es la ACCIÓN: la
calidad de cada candidato a largo lead ya está medida por
``real_fold_metrics`` (mismo pipeline fit/predict, rollout incluido, 4
ventanas trimestrales) — ``wmape_long`` por (modelo, propiedad). Esa es
la señal que decide el reemplazo, no una preferencia por defecto.

### D2 — Margen material: 0.95
Reemplazar tiene costo (heterogeneidad de modelos dentro del horizonte,
posible discontinuidad en lead 30). Solo se paga si el candidato es
≥5% mejor que el propio campeón a largo lead:
``wmape_long_alt < wmape_long_campeón × 0.95``. Con los datos actuales
de FIESTA: sarimax 10.2% < lightgbm 11.3% × 0.95 = 10.7% ✓ califica;
catboost 10.8% no. Si el campeón ya es el mejor largo, no hay cirugía
— el colapso distribucional no es un problema de precisión.

### D3 — Sin evidencia, no hay cirugía
Corridas sin fold metrics (sintéticas/muestra) o sin ``wmape_long``: el
colapso se registra en el manifiesto (``applied: false`` con la razón)
pero NO se reemplaza. El bug original era exactamente reemplazar a
ciegas; sin evidencia la acción por defecto es no tocar.

### D4 — El mecanismo es general, sin hardcode de FIESTA
Cualquier propiedad con campeón ML colapsado aplica la misma regla con
SUS folds. Si para otra propiedad pickup_curve es genuinamente el mejor
largo, la regla lo elige — no se excluye a la curva, solo deja de ser
el default. Filas ``PORTFOLIO`` de los folds no participan (se mira la
fila de la propiedad).

### D5 — Contrato del manifiesto extendido, no roto
``per_property_blend`` mantiene sus claves (trigger, short_model,
long_model, blend_split) y gana ``applied`` (bool) y ``long_selection``
(la evidencia numérica). Entradas con ``applied: false`` = colapso
detectado y campeón retenido: el consumidor del manifiesto ve la
decisión completa. El ``model_name`` de la tabla de forecast solo
declara blend cuando hay empalme real.

### D6 — El reemplazo se ajusta como candidato completo
El candidato elegido se entrena con el MISMO ``fit_candidate`` del
benchmark (no se reusan artefactos de fold): misma data, mismo corte.
Si el ajuste cae en cold-start (p. ej. curva sin historia suficiente),
el campeón se retiene y la razón queda en ``long_selection``.

## Risks / Trade-offs

- [Un candidato gana el largo por suerte de ventana] — mitigado: media de 4 folds + margen 5%; documentado que no se exige mayoría por fold (simplicidad sobre ingeniería extra).
- [Campeón retenido con rollout colapsado] — aceptado: los folds dicen que su largo sigue siendo el mejor disponible; el colapso distribucional queda visible en el manifiesto para monitoreo.
- [Discontinuidad en lead 30 entre dos modelos] — preexistente del blend original; el margen material reduce la frecuencia de empalmes injustificados.
- [Corridas de muestra sin blend] — cambio de comportamiento vs. el blend original (que empalmaba la curva a ciegas); documentado en D3 y cubierto por tests.
- [sarimax plano a largo lead] — su patrón DOW largo es plano (~337) contra reales con amplitud 107: pierde el patrón pero gana el nivel; los folds dicen que a largo lead eso ES mejor (10.2% vs 26.3%). El patrón corto lo aporta lightgbm en leads 1-30.
