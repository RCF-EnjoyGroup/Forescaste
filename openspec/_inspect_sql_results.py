"""Inspect the three SQL result files."""
import pandas as pd

FILES = [
    r"C:\Users\Rigoberto\Downloads\1. Estructura de snapshots.xlsx",
    r"C:\Users\Rigoberto\Downloads\2. Rango real de fechas y hoteles.xlsx",
    r"C:\Users\Rigoberto\Downloads\3. Profundidad del pickup histórico.xlsx",
]

for path in FILES:
    xl = pd.ExcelFile(path)
    print("\n" + "#" * 78)
    print(f"ARCHIVO: {path.split(chr(92))[-1]} | hojas: {xl.sheet_names}")
    print("#" * 78)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        print(f"\n--- Hoja '{sheet}' | filas: {len(df)} | columnas: {len(df.columns)} ---")
        print("Columnas:", df.columns.tolist())
        if len(df) <= 60:
            print(df.to_string())
        else:
            print(df.head(25).to_string())
            print(f"... ({len(df) - 25} filas mas)")
            # Distributions for large property tables
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    print(f"\nDescribe de {col}:")
                    print(df[col].describe().to_string())
                    print(f"Suma total de {col}: {df[col].sum():,.0f}")
