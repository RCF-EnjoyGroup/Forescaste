# Design — weekly-accuracy-grain

## Decisions

### D1 — El grano semanal es un desglose, no un reemplazo
La precisión diaria sigue siendo la métrica oficial del motor (benchmark,
selección, gate, KPIs). El grano semanal se AGREGA como desglose de
lectura: es la lente con la que Revenue opera (staffing/compras se
deciden por semana) y la única honesta para propiedades cuyo piso de
ruido diario es del tamaño de su señal (SJOSL: reales de 4 RN).

### D2 — Agregación por semana ISO con suma, no promedio
`metrics_by_property_weekly` suma forecast y reales por
(propiedad, semana) antes de medir. Sumar preserva el WMAPE ponderado
por volumen real (la semana con más RN pesa más) y es exactamente el
número que Revenue reconciliaría contra el PMS.

### D3 — La advertencia de MAPE es por evidencia de denominador, no por lista de propiedades
La tabla marca el MAPE con "†" cuando `p10_actual < 10` RN — es decir,
cuando el decil inferior de los días evaluados tiene reales tan chicos
que ±2 RN de error ya son "+50%". Es un criterio estadístico del propio
frame evaluado: SJOSL lo dispara hoy; una propiedad que crezca sale de
la advertencia sola. Sin hardcode de nombres.

### D4 — Investigados y descartados con evidencia (parte del hallazgo)
- **MARINA (sesgo nivel −13%)**: la demanda subió DESPUÉS del corte
  (jul 26.5 → oct 34.2 RN). Los folds muestran a lightgbm como el mejor
  disponible con bias neutral (−0.009); el sesgo es un límite de
  pronosticabilidad, no un defecto del modelo. Sin cambio.
- **LIREL (sobre-forecast entresemana)**: la hipótesis de cancelaciones
  queda descartada (booked cae de lead-60 a final solo en 13% de los
  stays; delta medio +16 RN). pickup_curve le gana al subcampeón por
  3pp de WMAPE en folds. Sin cambio.
- **Condicionamiento por mes de la curva (gap de octubre CORIN/VILLAS)**:
  A/B folds + T0 con tres variantes (mes-30, mes-15, split
  otoño/primavera). El mes arregla octubre-VILLAS (53.7→17.2) pero
  rompe octubre-FIESTA (+10) y octubre-MARINA (+20) — n=6 días con
  señales cruzadas; los folds son neutrales para CORIN/VILLAS y la única
  mejora fold-medible (curva MARINA −1.9 a −2.6pp) es sobre un modelo
  que NO opera en esa propiedad (su campeón es lightgbm). El impacto RN
  del gap de octubre es ~3% del error total. Sin cambio; prototipo
  revertido.

### D5 — El `p10_actual` entra al accuracy report como contexto de lectura
Columnas nuevas en `metrics_by_property`: `mean_actual`, `p10_actual`.
El consumidor del JSON puede replicar el criterio de advertencia sin
recomputar los reales.

## Risks / Trade-offs

- [Semanas parciales en los bordes de la ventana] — la primera/última semana del horizonte puede tener 6 días: se mide igual (la suma es correcta, el grano es consistente); documentado.
- [n_obs semanal ≈ 13 por propiedad] — suficiente para un desglose de lectura; el diagnóstico fino sigue en el grano diario.
- [Dos columnas de WMAPE pueden confundir] — la tabla titula cada una ("WMAPE" diaria vs "WMAPE sem") y la sección mantiene el orden por WMAPE diaria para no cambiar el ranking que el equipo ya conoce.
