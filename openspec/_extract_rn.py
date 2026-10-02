import json

nb = json.load(open("notebooks/hotel_roomnights_forecasting.ipynb", encoding="utf-8"))

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if any(
        k in src
        for k in (
            "=== Model Comparison", "Best model (by MAE)", "DM tests", "Total forecasted",
            "CONCLUSIONES — ROOM NIGHTS", "Capacity violation", "Days capped",
            "NULL-target rows excluded",
        )
    ):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                print("".join(out.get("text", []))[-2600:])
        print("-" * 70)
