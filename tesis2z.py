import pandas as pd
import numpy as np
import os

path_archivos = "/Users/camilajauregui/Documents/6to año/Tesis/datos fannie/"

# ── CARGA PANEL MENSUAL ────────────────────────────────────────────────────────
panel = pd.read_parquet(os.path.join(path_archivos, "panel_mensual.parquet"))
print(f"Panel mensual cargado: {panel.shape[0]:,} filas")

# ══════════════════════════════════════════════════════════════════════════════
# MÉTODO COHORT ANUAL
# Origen:  enero del año t
# Destino: enero del año t+1 si existe
#          si no → último registro disponible del préstamo dentro del año t
#          (captura defaults que ocurrieron entre febrero y diciembre
#           sin llegar a tener registro en enero del año siguiente)
#   Train: origen 2007 a 2019 inclusive
#   Test:  origen 2020, 2021 y 2022
# ══════════════════════════════════════════════════════════════════════════════

panel["anio"] = panel["fecha"].dt.year
panel["mes"]  = panel["fecha"].dt.month

# Filtrar solo años relevantes
panel = panel[panel["anio"].between(2007, 2023)].copy()

# ── ORIGEN: registro de enero de cada año ─────────────────────────────────────
enero = panel[panel["mes"] == 1].copy()
enero_orig = (
    enero
    .sort_values(["LOAN_ID", "fecha"])
    .groupby(["LOAN_ID", "anio"])
    .first()
    .reset_index()
    [["LOAN_ID", "anio", "grupo", "CSCORE_B", "ORIG_UPB", "CURR_RATE"]]
)

# ── DESTINO PRIMARIO: enero del año t+1 ───────────────────────────────────────
enero_dest = enero_orig[["LOAN_ID", "anio", "grupo"]].copy()
enero_dest["anio_origen"] = enero_dest["anio"] - 1
enero_dest = enero_dest.rename(columns={"grupo": "grupo_next", "anio": "anio_dest"})

corte_anual = enero_orig.merge(
    enero_dest[["LOAN_ID", "anio_origen", "grupo_next"]],
    left_on=["LOAN_ID", "anio"],
    right_on=["LOAN_ID", "anio_origen"],
    how="left"
).drop(columns=["anio_origen"])

# ── DESTINO FALLBACK: último registro dentro del año t ────────────────────────
sin_destino_mask  = corte_anual["grupo_next"].isna()
loans_sin_destino = corte_anual.loc[sin_destino_mask, "LOAN_ID"].unique()

if len(loans_sin_destino) > 0:
    panel_fallback = panel[
        (panel["LOAN_ID"].isin(loans_sin_destino)) &
        (panel["mes"] > 1)
    ][["LOAN_ID", "anio", "mes", "grupo"]].copy()

    fallback = (
        panel_fallback
        .sort_values(["LOAN_ID", "anio", "mes"])
        .groupby(["LOAN_ID", "anio"])
        .last()
        .reset_index()
        [["LOAN_ID", "anio", "grupo"]]
        .rename(columns={"grupo": "grupo_next_fallback"})
    )

    corte_anual = corte_anual.merge(fallback, on=["LOAN_ID", "anio"], how="left")

    usar_fallback = sin_destino_mask & corte_anual["grupo_next_fallback"].notna()
    corte_anual.loc[usar_fallback, "grupo_next"] = \
        corte_anual.loc[usar_fallback, "grupo_next_fallback"]

    corte_anual = corte_anual.drop(columns=["grupo_next_fallback"])

    n_primario = (~sin_destino_mask).sum()
    n_fallback = usar_fallback.sum()
    n_sin      = corte_anual["grupo_next"].isna().sum()
    print(f"\n  → Préstamos con destino enero t+1:             {n_primario:,}")
    print(f"  → Préstamos con fallback (último mes año t):   {n_fallback:,}")
    print(f"  → Préstamos sin destino (se dropean):          {n_sin:,}")

corte_anual = corte_anual.dropna(subset=["grupo", "grupo_next"]).copy()

print(f"\nTransiciones anuales válidas: {corte_anual.shape[0]:,}")

# ── CONTEOS POR AÑO ────────────────────────────────────────────────────────────
estados = ["S1_current", "S2_30dpd", "S3_60dpd", "S4_default"]

conteos_por_anio = {}
for anio in range(2007, 2023):
    sub = corte_anual[corte_anual["anio"] == anio]
    conteo = pd.crosstab(sub["grupo"], sub["grupo_next"])
    conteo = conteo.reindex(index=estados, columns=estados, fill_value=0)
    conteos_por_anio[anio] = conteo

# ── MATRIZ 2019 (base de comparación para proyecciones) ───────────────────────
N_2019 = conteos_por_anio[2019]
P_2019 = N_2019.div(N_2019.sum(axis=1), axis=0).fillna(0)

print("\n" + "="*60)
print("MATRIZ 2019  (base de comparación para proyecciones macro)")
print("="*60)
print(P_2019.round(4).to_string())

# ── VERIFICACIÓN ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MUESTRA DEL PANEL COHORT ANUAL (20 filas)")
print("="*60)
cols_mostrar = ["LOAN_ID", "anio", "grupo", "grupo_next",
                "CSCORE_B", "ORIG_UPB", "CURR_RATE"]
print(corte_anual[cols_mostrar].head(20).to_string(index=False))

print("\n" + "="*60)
print("TRANSICIONES A DEFAULT POR AÑO DE ORIGEN")
print("="*60)
resumen_anual = pd.DataFrame({
    anio: conteos_por_anio[anio]["S4_default"]
    for anio in range(2007, 2023)
}).T
resumen_anual.index.name = "anio_origen"
print(resumen_anual.to_string())

# ── GUARDAR ────────────────────────────────────────────────────────────────────
corte_anual.to_parquet(
    os.path.join(path_archivos, "panel_cohort_anual.parquet"), index=False
)
print("\nPanel cohort anual guardado como panel_cohort_anual.parquet")