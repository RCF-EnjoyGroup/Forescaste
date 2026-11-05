# Proposal — drift-level-correction

## Contexto (corrida real-20261104-1514, reales al 3-nov)

El punto por horizonte ya está DENTRO de los benchmarks de la industria en
todas las bandas de lead (7d 10.3%, 14-59d 16.5-16.7%, 60-89d 19.1%). El
peso del error ya no está en el horizonte sino en **sesgos de nivel por
hotel** que la capa de corrección por (propiedad × banda) no puede tocar:

| Hotel | Sesgo | Dirección | Cobertura banda |
|---|---|---|---|
| FIESTA | −11.5% | pronostica abajo (boom YoY) | 55.6% (sub-cubre) |
| CORIN | −10.0% | pronostica abajo | 73.3% |
| VILLAS | +19.8% | pronostica ARRIBA (régimen nuevo) | 93.3% |
| SJOSL | +17.1% | pronostica arriba | 70.0% |
| MARINA | +8.4% | arriba (hotel chico) | 68.9% |

La corrección de bias existente aplicó solo 11 de 42 celdas: sus gates
(como deben) rechazan sesgos que giran entre folds del año. Pero un
**drift de nivel reciente** no es un bias del año: es un régimen que
empezó hace semanas. La señal correcta es la que ya producimos para el
monitoreo: el **tracking signal del forecast PUBLICADO** — el error del
artefacto que el negocio vio, contra reales que iban llegando.

## Propuesta

Nueva capa de corrección de nivel, POR PROPIEDAD (no por banda), estimada
exclusivamente de los errores del forecast publicado previo (walk-forward,
cero lookahead):

1. **Señal**: errores del forecast publicado más reciente graduable
   (`vintage_detail`), sobre la ventana reciente de estancias
   (`window_days`, default 60): `rel = Σ(fc − act) / Σ|act|` y
   `ts = Σerr / media|err|` (misma definición TS del monitoreo).
2. **Peso suave** (sin estado, anti-parpadeo): `w = min(|ts| / trigger, 1)`
   — a TS=4 corrige completo; a TS=2, a medias; a TS=1, un cuarto. Nada
   de interruptores on/off que oscilen con el umbral.
3. **Disciplina**: shrinkage `n/(n+shrinkage)`, clip ±`max_rel` (±20%),
   `min_obs` para activar, correcciones <0.5% no tocan el punto.
4. **Aplicación**: multiplicativa al punto, ANTES de la corrección por
   banda y del piso de reservas/techo de capacidad; el dataset publicado
   gana la columna `level_correction_factor` y el manifiesto publica el
   detalle por propiedad (n, rel, ts, peso, factor, razón) — nada
   silencioso.
5. **Modo**: SOLO en corridas `futuro` (la publicación operativa). En modo
   verificación el horizonte ya tiene reales y corregirlo contaminaría la
   medición — la evaluación honesta queda intacta.

Bajo knob `monitoring.level_correction` — el default lo decide la
validación A/B, no a priori.

## Validación (obligatoria antes del default)

**Replay walk-forward sobre la historia de corridas futuras del registro**
(pares publicación anterior → publicación siguiente, en orden de tiempo):
el factor se estima SOLO con errores ≤ corte de la corrida siguiente y se
aplica a su horizonte publicado; se puntúa contra los reales
materializados (data vigente, revisada — la comparación es pareada así que
la revisión afecta a baseline y corregido por igual):

1. WMAPE global del conjunto de añadas mejora (≥2% relativo);
2. FIESTA: |sesgo| mejora materialmente (≥5pp hacia 0);
3. ninguna propiedad empeora WMAPE >2pp;
4. la cobertura de banda no empeora >5pp en ninguna propiedad.

## Non-goals

- No tocar la selección de campeón, el blend ni los gates existentes.
- No aplicar en modo verificación (honestidad de la evaluación).
- No corregir por banda con esta capa: la banda existente sigue a cargo
  de la calibración conformal; el nivel es por propiedad.

## Validación A/B — RESULTADO NEGATIVO (decisión documentada)

Replay walk-forward sobre el registro de corridas futuras
(`scripts/ab_level_correction.py`), dos configuraciones:

1. **Replay completo (10 pares, 545 estancias)** — incluye publicaciones
   pre-corrección de alcance MARINA (artefacto de mezcla de alcance en
   el baseline, no de la regla). Cuando la regla disparó, la corrección
   llegó a regiones YA calibradas: FIESTA sesgo −0.4% → −1.0% y
   cobertura 90.8% → 81.7% (sobre-corrección; criterios 1, 2 y 4 fallan).
2. **Replay post-corrección (3 pares, 50 estancias)** — solo pares con
   ambos runs después del 27-oct: la regla prácticamente no dispara
   (factores vacíos en 2 de 3 pares). El punto publicado a leads cortos
   ya está calibrado por el piso de reservas (FIESTA +0.2% de sesgo,
   cobertura 100%).

**Causa raíz del resultado**: la señal (errores de la publicación previa)
solo observa la región YA materializada de esa publicación — estancias de
los últimos días, leads cortos, donde el forecast publicado es bueno
porque lleva las reservas en libros. El drift que motivó el change
(FIESTA −11.5% en la revisión) vive en leads 14-59, cuya región de una
publicación materializa 2-4 semanas después — la señal no lo ve a
tiempo, y cuando lo ve, la publicación nueva ya lo absorbió parcialmente.

**Decisión: `level_correction.enabled: false` por default.** La
maquinaria queda disponible como knob con la evidencia documentada
(precedente: `seasonal_cells`, `interval_recency_decay`). Vía de re-apertura
cuando el registro acumule historia: variante con señal de la publicación
de hace ~45-60 días (su región de leads medios ya materializada) aplicada
solo a la región de leads medios del horizonte nuevo — hoy no es
validable (no existe esa historia; el registro arranca el 09-oct).

Lo que SÍ queda operativo de este change: el manifiesto ahora expone
`level_correction` con el estado del knob y la razón cuando no aplica,
y la alerta de drift/TS existente sigue siendo la señal de acción
(revisión manual documentada del override, no corrección automática).
