# Guía para Sistemas — Exportación de Agregados para Forecasting

**Para:** equipo de sistemas / base de datos
**Proyecto:** Pronóstico de Room Nights — Enjoy Costa Rica
**Versión:** 2026-10-07 (v0.4.0)

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
```

---

## Dónde colocar los archivos

En la carpeta del proyecto (o ajustar las rutas en `config.yaml` → `roomnights_real.csv_paths`):

```
data/roomnights_real/
├── demand_daily.csv
├── pickup_by_lead.csv
└── capacity_daily.csv
```

---

## Cadencia sugerida

| Agregado | Cadencia | Por qué |
|---|---|---|
| Demanda (Query 1) | Diaria | Cada día se cierra una nueva fecha de estadía |
| Pickup (Query 2) | Diaria | Los libros cambian todos los días (reservas/cancelaciones) |
| Capacidad (Query 3) | Semanal o al cambio | El inventario oficial cambia poco (renovaciones/cierres) |

> Nota: el pickup **incluye fechas de estadía futuras** — eso es correcto y
> necesario: son las reservas vigentes que alimentan el pronóstico.

---

## Notas importantes

1. **NO unificar nombres de propiedades.** El pipeline canoniza solo las
   variantes (`CORIN`, `Hotel Royal Corin`, `LAPAS`, `SJOSL`, etc.) mediante el
   diccionario en `config.yaml → roomnights_real.property_map`. Si aparece una
   propiedad **nueva**, el pipeline falla ruidosamente a propósito (mejor un error
   visible que un hotel sin capacidad). En ese caso: avisar a analítica para
   agregarla al mapeo.

2. **`snap_flag` es la columna semántica clave**: 0 = realizado, 1 = libros
   futuros. El Query 1 filtra 0 y el Query 2 filtra 1. Si el significado de esa
   columna cambiara en el ETL, avisar — es un cambio de contrato.

3. **`SYSTEM_ADJUST` se excluye de la demanda** (cargos de servicio, siempre
   rooms=0). Si aparecen otros ratecodes de ajuste contable, comunicarlo.

4. **Volumen esperado**: demanda ~8 propiedades × historial (miles de filas,
   liviano); pickup es más grande pero ya agregado (manejable como export CSV
   directo desde pgAdmin/DBeaver — Excel puede quedarse corto).

5. **Los NULL de capacidad se dejan como NULL**: el pipeline aplica un fallback
   documentado (último valor conocido por propiedad) y lo reporta. No rellenar
   con ceros — un cero rompería el techo de ocupación.

---

## Verificación rápida post-exportación

```text
□ demand_daily.csv   → sin duplicados de (property, stay_date)
□ pickup_by_lead.csv → lead_days siempre > 0
□ capacity_daily.csv → una fila por (property, business_date)
□ Las 8 propiedades aparecen en demanda: LAPAS, VILLAS, SJOSL,
  Hotel Royal Corin, LIRAK, LIREL, MARINA, FIESTA
□ pickup_by_lead.csv contiene fechas de estadía FUTURAS (correcto)
```

---

## Qué hace analítica con los 3 archivos

Ejecuta `notebooks/hotel_roomnights_realdata.ipynb` (sin cambios de código):

1. Valida, canoniza propiedades y construye las series diarias
2. Compara 8 modelos con protocolo honesto (multi-paso real, baseline naive,
   backtesting de 6 ventanas, test de significancia DM, bias)
3. Pronostica 90 días del portafolio **y por propiedad**, con:
   - reservas actuales como señal (`rotb_d90/180/365` — correlación 0.93 con lo realizado)
   - techo de capacidad del inventario y ocupación implícita
   - bandas de incertidumbre conformales con cobertura reportada
   - hotels nuevos (cold-start) estimados desde sus libros, con flag explícito
   - **cobertura de libros**: qué % del pronóstico ya está reservado

**Contacto**: equipo de analítica — cualquier duda sobre las queries o el
formato, antes de reintentar a ciegas: el contrato está en este documento y en
`src/forecasting/data/real_sources.py` (fuente de verdad del código).
