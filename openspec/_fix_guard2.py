"""Guard fix: short-history -> recent-level anchoring (BooksAnchor invalid on ramps)."""
import ast
import json

p = "notebooks/hotel_roomnights_realdata.ipynb"
nb = json.load(open(p, encoding="utf-8"))


def to_src(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


for c in nb["cells"]:
    if c["cell_type"] == "code" and "GUARD short-history" in "".join(c["source"]):
        src = "".join(c["source"])
        old = """    else:
        cover = pick_g[rotb_cols_ml[0]].reindex(h_test_dates).notna().mean() \\
            if len(h_test_dates) else 0.0
        if cover >= 0.5:
            per_prop_model[prop] = 'BooksAnchor (guard short-history)'
            pred_fn = lambda h, d: books_anchor_prop(pick_g, h, d)
            bh = pick_g[rotb_cols_ml[0]].reindex(pre.index)
            wb = pd.DataFrame({'y': pre.values, 'b': bh.values}).dropna().tail(120)
            if len(wb) > 7:
                comp_g = float((wb['y'] / wb['b']).median())
                short_pools[prop] = (wb['y'].values - wb['b'].values * comp_g)[-60:]
            else:
                short_pools[prop] = (pre.values[7:] - pre.values[:-7])[-60:]
        else:
            per_prop_model[prop] = 'SeasonalNaive (guard short-history)'
            pred_fn = lambda h, d: sn7_preds(h.values, len(d))
            short_pools[prop] = (pre.values[7:] - pre.values[:-7])[-60:] \\
                if len(pre) > 7 else np.zeros(1)
        print(f"{display_name.get(prop, prop)}: GUARD short-history "
              f"({len(pre)}d pre-test, libros {cover:.0%}) -> {per_prop_model[prop]}")"""
        new = """    else:
        # Short-history guard: recent-level anchoring (weekly tiling). BooksAnchor
        # is NOT valid here: completion-ratio stationarity is unverifiable with
        # <120d and ramps violate it (SJO's opening-period realized/books_d90 ~8x
        # vs portfolio 2.68x — its booking curve is extremely back-loaded and the
        # ratio applied to future books gave an implausible ~4x-level forecast).
        per_prop_model[prop] = 'SeasonalNaive (guard short-history)'
        pred_fn = lambda h, d: sn7_preds(h.values, len(d))
        short_pools[prop] = (pre.values[7:] - pre.values[:-7])[-60:] \\
            if len(pre) > 7 else np.zeros(1)
        print(f"{display_name.get(prop, prop)}: GUARD short-history "
              f"({len(pre)}d pre-test) -> {per_prop_model[prop]}")"""
        assert old in src, "bloque guard no encontrado"
        src = src.replace(old, new, 1)
        c["source"] = to_src(src)
        c["outputs"] = []
        c["execution_count"] = None
        break
else:
    raise AssertionError("celda per-prop no encontrada")

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        try:
            ast.parse("".join(c["source"]))
        except SyntaxError as e:
            raise AssertionError((i, str(e)[:100]))

json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("guard revertido a nivel reciente (SN7) con justificacion documentada; sintaxis OK")
