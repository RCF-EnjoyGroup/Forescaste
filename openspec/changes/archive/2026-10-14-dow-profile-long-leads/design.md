# Design — dow-profile-long-leads

## Decisions

### D1 — Nivel del mejor modelo × patrón de la historia
El blend accuracy-aware elige el reemplazo largo por EVIDENCIA de folds
(`wmape_long`). Los modelos stats (SARIMAX/ETS) ganan ese ranking porque
su nivel es estable — pero a 60-90 pasos su componente estacional AR
decae y el forecast converge a plano. El patrón DOW de una propiedad es
un hecho operativo (el hotel tiene su ritmo semanal) conocible a
cualquier lead desde historia ≤ corte. La composición
``nivel_ganador × perfil_DOW`` conserva la ventaja del nivel (lo que la
evidencia midió) y restaura lo que el modelo no puede extrapolar.

### D2 — Perfil de 84 días, normalizado, con guardas
Ventana de 12 semanas: suficiente para ~12 obs por DOW, suficientemente
reciente para seguir el régimen. Normalización a media 1 sobre los 7
factores — el nivel del segmento empalmado no se desplaza (el mix DOW de
un tramo de 60 días es aproximadamente uniforme). Guardas: < 28 días de
historia o < 3 obs por DOW → sin perfil (plano), registrado en el
manifiesto. La amplitud del perfil reciente puede exagerar la del
tramo objetivo (12 sem vs la estación del stay): la validación mostró
que aun con amplitud imperfecta el patrón domina al plano en las 5
ventanas.

### D3 — Solo al segmento empalmado, solo con modelos planos
El perfil aplica únicamente cuando el blend empalma un reemplazo de
kind `stats` en leads > 30 (hoy: FIESTA con sarimax). El tramo corto
(≤ 30) es el rollout del campeón ML con sus lags reales — no se toca.
Los campeones stats sin blend quedan fuera: sin evidencia validada para
ese caso, y cambiarlos es otro cambio.

### D4 — El perfil NO participa en la selección
`_select_long_replacement` sigue decidiendo por `wmape_long` puro sobre
modelos sin perfil. Aplicar el perfil en la selección sería optimizar
el ranking hacia un candidato con perfil (result-shopping); la validación
se hizo exactamente sobre la configuración que se opera: primero se
elige el reemplazo, después se le aplica el patrón.

### D5 — Declaración en el manifiesto
`per_property_blend` gana `long_pattern`: `"dow_profile(84d)"` cuando
aplicó, `"flat"` cuando el guard lo desactivó (historia corta). El
consumidor ve la decisión completa: quién opera cada tramo y con qué
patrón.

## Risks / Trade-offs

- [Perfil reciente vs estación del tramo] — un perfil de 12 semanas puede no representar la estación del stay lejano; la validación (5 ventanas, −1.9pp promedio, nunca daña) cubre exactamente este riesgo en los datos reales. Si una futura propiedad lo rompe, el fold de selección lo detectará (el reemplazo stats con perfil compite en folds).
- [Amplitud exagerada] — el perfil reciente de FIESTA (jue 0.74, sab 1.17) es más empinado que el patrón observado del tramo; aun así el wmape mejora. No se suaviza el perfil (tuning ad hoc sin evidencia); documentado.
- [Cambios de nivel por mix DOW] — la normalización es sobre los 7 factores, no sobre el tramo; un tramo con mix DOW sesgado desplaza el nivel ±2-3%; despreciable frente al gap de picos que corrige.
