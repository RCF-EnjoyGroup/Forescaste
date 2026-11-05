# Design — bias-correction-mode

## Kill-switch (`bias_correction.enabled`)

En `run_end_to_end`, si `enabled: false`: no se estiman factores
(`lead_bias_factors` no se llama), no se aplican (`apply_bias_factors`
no se llama), y el manifiesto publica `bias_correction: {enabled: false}`.
Las bandas (`operative_residual_bands`) se calculan con `bias_factors={}`
— calibran el pipeline que se publica, que ahora va sin corrección. El
resto de la corrida (campeón, blend, restricciones, intervalos) intacto.

## Ponderación de folds por recencia (`fold_recency_decay`)

En `lead_bias_factors`, cada observación recibe
`w = decay ** (max_fold − fold)` (el fold de índice máximo es el más
reciente — mismo convenio de `operative_residual_bands`). Con
`decay = 1.0` todos los pesos son 1.0 → **bit a bit el comportamiento
actual** (test de regresión).

- `rel` ponderado: `Σ w·(fc−act) / Σ w·|act|` — la magnitud del sesgo la
  domina el régimen reciente;
- **consistencia como votación ponderada**: cada fold vota el signo de su
  bias relativo y su voto vale `w_fold`; `consistency = Σ w_f·[signo_f =
  signo_agg] / Σ w_f`. Un fold viejo que discrepa descuenta menos que hoy;
- `min_obs` sigue contando observaciones crudas (evidencia, no magnitud);
- shrinkage y clip sin cambios.

El estimador compartido `_bias_cell_estimate` recibe el decay y lo aplica
a celdas de banda y estacionales por igual (la celda estacional hereda el
mismo convenio si algún día se re-activa).

## A/B de 3 brazos (misma data, reales al 3-nov)

- A: `real-20261104-1514` (ya existe — capa ON, folds uniformes).
- B: re-corrida con `enabled: false`.
- C: re-corrida con `fold_recency_decay: 0.7`.
Comparación pareada sobre `powerbi/forecast.csv` (punto publicado vs
reales) y `forecast_accuracy_report.json` (cobertura): sesgo y WMAPE por
propiedad, cobertura por propiedad. Script reproducible que imprime la
tabla de 3 brazos y el veredicto contra los criterios.

## Riesgos

- **Apagar la capa pierde su valor en regímenes estables**: si el brazo B
  gana hoy, el knob permite re-activarla cuando el portafolio se
  estabilice; la decisión queda documentada con la fecha y la evidencia.
- **Decay demasiado agresivo**: 0.7 con 4 folds da pesos 1/0.7/0.49/0.34 —
  el fold reciente domina sin anular a los demás. El A/B mide si eso
  acerca el sesgo a 0; si sobre-corrige, el clip ±30% y la consistencia
  ponderada lo acotan.
- **Retrocompatibilidad**: ambos knobs default al comportamiento actual
  (`enabled: true`, `fold_recency_decay: 1.0`) — sin knobs, bit a bit.
