# Guía para Sistemas — Exportación de Agregados para Forecasting

**Para:** equipo de sistemas / base de datos
**Proyecto:** Pronóstico de Room Nights — Enjoy Costa Rica
**Versión:** 2026-10-10 (v0.5.0)

> v0.5.0 — sincronizado con el código: rutas reales de `config.yaml`
> (`data.sources`), comando de ejecución `python -m src.forecasting.run_real`
> (con los modos verificación y `--futuro`) y checklist ajustado a lo que el
> loader realmente valida.

---

## Resumen en una frase

El pipeline de pronóstico **no accede a las tablas crudas** — las ~14M de filas de
snapshots se quedan en la base. Sistemas ejecuta **3 queries de agregación** y
exporta **3 archivos CSV**; el notebook de analítica consume esos archivos sin
ningún cambio de código.

---

## Las 3 Queries

Ejecutar cada query y exportar el resultado como CSV (UTF-8, fechas `YYYY-MM-DD`).

### Query 1 — Demanda realizada (el TARGET del pronóstico)

Exportar como **`demand_daily.csv`**:

```sql
SELECT
    property,
    stay_date,
    SUM(rooms) AS rooms_sold,
    SUM(CAST(REPLACE(room_revenue, ',', '.') AS DOUBLE PRECISION)) AS room_revenue
FROM staging.enjoy_fac_hotel
WHERE snap_flag = 0
  AND COALESCE(ratecode, '') NOT IN ('SYSTEM_ADJUST')
  AND rooms IS NOT NULL
GROUP BY property, stay_date
ORDER BY property, stay_date;
```

### Query 2 — Reservas en libros por anticipación (el PICKUP)

Exportar como **`pickup_by_lead.csv`**:

```sql
SELECT
    property,
    stay_date,
    (stay_date - snapshotdate) AS lead_days,
    SUM(rooms) AS rooms_booked
FROM staging.enjoy_fac_hotel
WHERE snap_flag = 1
  AND rooms IS NOT NULL
GROUP BY property, stay_date, (stay_date - snapshotdate)
ORDER BY property, stay_date, lead_days;
```

### Query 3 — Capacidad oficial (el TECHO / ocupación)

Exportar como **`capacity_daily.csv`**:

```sql
SELECT
    property,
    business_date,
    SUM(oficial_inventory) AS available_rooms
FROM public.enjoy_inventory
GROUP BY property, business_date
ORDER BY property, business_date;
```

### Fuente 4 (OPCIONAL) — Eventos privados (conciertos, congresos, torneos)

Exportar como **`data/events.csv`** (ruta configurable en `config.yaml` →
`data.events_path`). Es la única fuente que NO viene de la base de
transacciones: son los eventos que el negocio conoce por adelantado
(calendario de eventos de Revenue Management, calendario deportivo/cultural
de la zona). **Mientras no exista el archivo, el motor funciona igual y
declara la fuente ausente en el manifiesto.**

```
event_name,event_date,property
Congreso Tech CR,2026-05-10,CORIN          ← evento SOLO de esa propiedad
Festival Internacional de Música,2026-09-12,  ← blank/NaN = TODO el portafolio
```

Reglas del loader (falla la corrida antes de entrenar si no se cumplen):

| Regla | Detalle |
|---|---|
| Columnas | `event_name` (o `evento`/`nombre`), `event_date` (o `fecha`), `property` opcional (o `propiedad`/`hotel`) |
| Fechas | `YYYY-MM-DD`; una fecha ilegible **detiene la corrida** con el número de filas afectadas |
| Duplicados | (nombre + fecha + propiedad) — **detienen la corrida** con ejemplos |
| `property` | Nombre canónico (mismos nombres que en `demand_daily.csv`); vacío/NaN = evento de portafolio |
| Alcance temporal | Eventos futuros Y pasados: los pasados alimentan las curvas de pickup por ventana de evento (dentro de la ventana de recencia de 3 años) |

Qué hace el motor con cada evento: su fecha entra a la feature de distancia
al "feriado/evento más cercano" de los candidatos ML, y su ventana ±4 días
genera una curva de pickup propia (con fallback si hay poca historia). Un
evento con `property` asignada solo afecta a esa propiedad.

---

## Formato de exportación

| Regla | Detalle |
|---|---|
| Encoding | UTF-8, separador coma, sin BOM |
| Fechas | `YYYY-MM-DD` |
| Nombres de columnas | **EXACTOS** como en cada query (el loader valida) |
| `room_revenue` | El query ya lo convierte a número con punto decimal — **no reformatear después** |
| Sin filas extra | Ni totales, ni encabezados duplicados, ni filas de notas |

**Columnas esperadas por archivo:**

```
demand_daily.csv   → property, stay_date, rooms_sold, room_revenue
pickup_by_lead.csv → property, stay_date, lead_days, rooms_booked
capacity_daily.csv → property, business_date, available_rooms
events.csv (opcional) → event_name, event_date, property (opcional)
```

---

## Dónde colocar los archivos

En la carpeta `data/` de la raíz del proyecto (las rutas por defecto ya están
en `config.yaml` → `data.sources`; solo hay que ajustarlas si los archivos
viven en otro lado):

```yaml
# config.yaml
data:
  sources:
    capacity: data/capacity_daily.csv
    demand:   data/demand_daily.csv
    pickup:   data/pickup_by_lead.csv
```

```
data/
├── demand_daily.csv
├── pickup_by_lead.csv
├── capacity_daily.csv
└── events.csv   ← OPCIONAL: sin este archivo el motor funciona igual
```

> `data/` está en `.gitignore`: los CSV reales nunca se commitean.

---

## Cadencia sugerida

| Agregado | Cadencia | Por qué |
|---|---|---|
| Demanda (Query 1) | Diaria | Cada día se cierra una nueva fecha de estadía |
| Pickup (Query 2) | Diaria | Los libros cambian todos los días (reservas/cancelaciones) |
| Capacidad (Query 3) | Semanal o al cambio | El inventario oficial cambia poco (renovaciones/cierres) |
| Eventos (Fuente 4, opcional) | Al conocerse / al cambio | Un evento nuevo vale más mientras más lead tenga |

> Nota: el pickup **incluye fechas de estadía futuras** — eso es correcto y
> necesario: son las reservas vigentes que alimentan el pronóstico.

---

## Notas importantes

1. **NO unificar nombres de propiedades.** El pipeline canoniza solo las
   variantes (`CORIN`, `Hotel Royal Corin`, `LAPAS`, `SJOSL`, etc.) mediante el
   diccionario en `config.yaml → data.property_aliases`. Si aparece una
   propiedad **nueva**, el pipeline falla ruidosamente a propósito (mejor un error
   visible que un hotel sin capacidad). En ese caso: avisar a analítica para
   agregarla al mapeo.

2. **`snap_flag` es la columna semántica clave**: 0 = realizado, 1 = libros
   futuros. El Query 1 filtra 0 y el Query 2 filtra 1. Si el significado de esa
   columna cambiara en el ETL, avisar — es un cambio de contrato.

3. **`SYSTEM_ADJUST` se excluye de la demanda** (cargos de servicio, siempre
   rooms=0). Si aparecen otros ratecodes de ajuste contable, comunicarlo.

4. **Volumen observado** (referencia): demanda ~14k filas (8 propiedades,
   historial desde 2013); capacidad ~40k filas; pickup ~1.5M filas ya agregado
   — exportar como CSV directo desde pgAdmin/DBeaver, **Excel se queda corto**
   para el pickup.

5. **Los NULL de capacidad se dejan como NULL**: el pipeline aplica un fallback
   documentado (último valor conocido por propiedad) y lo reporta. No rellenar
   con ceros — un cero rompería el techo de ocupación.

---

## Verificación rápida post-exportación

```text
□ demand_daily.csv   → sin duplicados de (property, stay_date)
□ pickup_by_lead.csv → lead_days >= 0 (negativo = error del loader)
□ capacity_daily.csv → una fila por (property, business_date)
□ Las 8 propiedades aparecen en demanda: LAPAS, VILLAS, SJOSL,
  Hotel Royal Corin, LIRAK, LIREL, MARINA, FIESTA
  (en capacidad pueden venir como CORIN — el alias se aplica solo)
□ pickup_by_lead.csv contiene fechas de estadía FUTURAS (correcto)
□ events.csv (si existe) → fechas YYYY-MM-DD legibles y sin duplicados
  (nombre + fecha + propiedad)
```

---

## Qué hace analítica con los 3 archivos

Ejecuta el motor sobre los CSV (sin cambios de código). Son **dos modos
distintos y excluyentes** — elegir uno según el objetivo:

```bash
# (A) VERIFICACIÓN: retrocede el corte 90d y pronostica una ventana que YA
# tiene reales, para medir el error real del motor (MAPE/WMAPE). No sirve
# como pronóstico operativo: su ventana termina en el último real.
python -m src.forecasting.run_real

# (B) PRONÓSTICO OPERATIVO: corte = último real y pronostica lo que falta
# (hasta 31-dic o la fecha indicada en --hasta). Éste es el que usa el negocio.
python -m src.forecasting.run_real --futuro
python -m src.forecasting.run_real --futuro --hasta 2027-03-31
```

Salida: `outputs/runs/real-<fecha>/` (modo A) o `outputs/runs/futuro-<fecha>/`
(modo B), cada uno con los 10 entregables + `run_manifest.json`. En el modo B
la ventana aún no tiene reales, así que métricas y banda provienen de un
backtest real previo (etiquetado `metrics_source: injected_backtest` en el
manifiesto) — no se inventan reales futuros.

1. Valida, canoniza propiedades y construye las series diarias
2. Compara 7 candidatos + baseline con protocolo honesto (backtest expanding,
   4 folds trimestrales × el horizonte operativo completo (90d, config
   `benchmark`) sobre la serie agregada, selección de campeón por evidencia
   out-of-sample, desgloses por lead/propiedad/temporada)
3. Pronostica el horizonte configurado del portafolio **y por propiedad**, con:
   - reservas actuales como señal (`rotb_d90/180/365` — correlación 0.93 con lo realizado)
   - techo de capacidad del inventario y ocupación implícita
   - bandas de incertidumbre conformales con cobertura reportada
   - hotels nuevos (cold-start) estimados desde sus libros, con flag explícito
   - **cobertura de libros**: qué % del pronóstico ya está reservado

**Contacto**: equipo de analítica — cualquier duda sobre las queries o el
formato, antes de reintentar a ciegas: el contrato está en este documento y en
`src/forecasting/run_real.py` (`load_real_sources`) + `config.yaml → data`
(fuente de verdad del código).
