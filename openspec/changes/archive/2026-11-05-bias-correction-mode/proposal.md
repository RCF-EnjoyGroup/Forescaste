# Proposal — bias-correction-mode

## Contexto (diagnóstico de la corrida real-20261104-1514)

El peso del error ya no está en el horizonte (todas las bandas de lead
dentro de benchmarks de industria) sino en sesgos de nivel por hotel. El
diagnóstico celda por celda del horizonte publicado reveló que la capa de
corrección de bias por (propiedad × banda) — que usa el PROMEDIO DEL AÑO
de folds — está desalineada con el régimen reciente, en ambas direcciones:

| Hotel·banda | La corrección hizo | El raw del horizonte era | Veredicto |
|---|---|---|---|
| MARINA 30-59 | −15.6% | −2% (casi perfecto) | DAÑÓ −15pp |
| MARINA 7-13 | +9.8% | +44% arriba | dirección equivocada |
| CORIN 14-29 | −5.4% | −24% | ayudó poco (drift mayor) |
| SJOSL 30-59 | −6.9% | +8.8% → +1.3% publicado | AYUDÓ |
| FIESTA 30-59 | nada (gate bloqueó) | −13.5% | drift sin tocar |

La premisa original de este frente ("shrinkage débil para hoteles
chicos") NO es lo que la data muestra: el problema es que el promedio
anual de folds mezcla regímenes, y celdas consistentes EN EL AÑO cargan
sesgos que ya no coinciden con el régimen de las últimas semanas.

## Propuesta

Dos knobs sobre la capa existente (sin tocar gates, escalera ni bandas):

1. **`bias_correction.enabled`** (kill-switch): permite apagar la capa
   completa. Hoy es incondicional — si el promedio anual está dañando
   neto, no hay forma de desactivarla sin código.
2. **`bias_correction.fold_recency_decay`** (default 1.0 = comportamiento
   actual bit a bit): pondera los folds por recencia en el estimador —
   el fold más reciente pesa 1.0, el anterior decay, etc. Alinea la
   evidencia con el régimen (el mismo patrón que `interval_recency_decay`
   ya aplica a las bandas). La consistencia de signo pasa a ser votación
   ponderada por fold.

## Validación A/B de 3 brazos — RESULTADO: defaults sin cambio (decisión documentada)

Misma data (reales al 3-nov), tres brazos pareados sobre el horizonte de
verificación: **A** capa ON uniforme (`real-20261104-1514`), **B** capa
OFF (`real-20261105-1442`), **C** capa ON con `fold_recency_decay: 0.7`
(`real-20261105-1534`).

| Propiedad | A sesgo | B sesgo | C sesgo | A wmape | B wmape | C wmape |
|---|---|---|---|---|---|---|
| CORIN | −10.0% | −9.2% | −9.3% | 22.3 | 21.8 | 21.9 |
| FIESTA | −11.5% | −11.3% | −11.5% | 13.0 | 12.9 | 13.0 |
| LIREL | +2.7% | +0.8% | +2.6% | 14.9 | 14.7 | 14.8 |
| MARINA* | +8.4% | +2.6% | +3.2% | 36.1 | 35.1 | 35.4 |
| SJOSL | +17.1% | **+21.8%** | +17.1% | 43.6 | **45.4** | 43.6 |
| VILLAS | +19.8% | +19.8% | +19.8% | 24.5 | 24.5 | 24.5 |

\* MARINA cambió de modelo entre brazos por la cadena de histéresis del
registro (`futuro-20261104-1525` giró el incumbente a sarimax; los brazos
B/C lo heredaron) — su mejora NO es atribuible a la capa.

**Hallazgos:**
1. **En SJOSL la capa AYUDA**: sin ella el sesgo empeora +4.7pp y el
   WMAPE +1.8pp. La corrección 30-59 (raw +8.8% → publicado +1.3%) es
   real y valiosa.
2. **En MARINA el efecto es confundido** por el cambio de modelo — la
   comparación limpia B vs C (ambos sarimax) muestra diferencias
   pequeñas entre capa OFF y recencia (2.6% vs 3.2%).
3. **CORIN/LIREL mejoran ~1pp sin la capa** (celdas marginales), y
   FIESTA/VILLAS son intocadas (sus celdas están bloqueadas por gates o
   el drift no está en los folds).
4. **La variante recencia 0.7 no daña a nadie** (SJOSL idéntico a A,
   resto mejor-o-igual, |sesgo| medio foco 10.3% vs 11.7% en A) pero no
   cumple el criterio de mejora estricta (SJOSL empata, FIESTA empata,
   sesgo global RN-ponderado −5.0% vs −4.9%).

**Decisión: defaults sin cambio** — `enabled: true`,
`fold_recency_decay: 1.0`. El criterio 2 falló en SJOSL/FIESTA en ambos
brazos. La recencia 0.7 queda como knob documentado (sin daño medido)
para re-visitar cuando el portafolio esté en transición más marcada.

**Lección de fondo:** el sesgo dominante del horizonte (FIESTA −11.5%,
VILLAS +19.8%, SJOSL +64% en 60-89) es **drift de régimen**, no bias de
folds — la capa con cualquier ponderación no lo alcanza. La vía sigue
siendo la alerta TS/PSI + revisión de negocio (drift-level-correction,
archivado, documentó lo mismo desde la señal publicada).

## Non-goals

- No tocar gates, escalera de fallback ni la calibración conformal de bandas.
- No corregir el drift de régimen (FIESTA boom) automáticamente: la
  evidencia del change drift-level-correction (archivado, A/B negativo)
  mostró que la señal publicada no llega a tiempo; la alerta TS/PSI y la
  revisión de negocio siguen siendo la vía.
