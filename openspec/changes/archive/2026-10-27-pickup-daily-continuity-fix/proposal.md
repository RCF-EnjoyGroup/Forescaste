# Proposal — pickup-daily-continuity-fix

## Contexto

La revisión openspec del 27-oct rastreó los avisos de "fechas faltantes" del
DQ report (VILLAS 332, CORIN 213, SJOSL 71, FIESTA 66): provienen de la
fuente PICKUP, no de demanda. La demanda está limpia (VILLAS 0 huecos
reales; solo LIREL tiene su gap de COVID, 93 días, correctamente reportado).

## Problema

El chequeo `missing_dates` exige continuidad diaria a TODAS las fuentes por
igual. Para demanda y capacidad eso es correcto (son series físicas diarias:
cada día existe un real y un inventario). Para el pickup es un falso
positivo por diseño: una fecha de estancia **sin reservas legítimamente no
trae fila** — un stay futuro con cero on-the-books no es un hueco de
calidad, es un día vacío. El chequeo genera 4 de las 5 advertencias actuales
del DQ report, todas ruido — y ruido en un gate de calidad erosiona la
confianza en las advertencias reales (la única real es el gap COVID de
LIREL).

## Propuesta

`check_missing_dates` respeta la semántica de la fuente: exige continuidad
diaria a series físicas (demanda, capacidad) y la omite para el pickup
(registro disperso de eventos). La omisión queda documentada en el propio
reporte (nota por fuente), no silenciosa.

## Non-goals

- Detectar huecos en el rango "snapshot activo" del pickup (ventana donde sí
  debería haber filas por fecha de extracción): item separado si Sistemas
  exporta extracciones espaciadas de forma irregular.
- Validez de leads/snapshots del pickup: ya cubierta por sus checks
  propios (`invalid_lead`, `duplicate_snapshots`, `pickup_inconsistency`).
