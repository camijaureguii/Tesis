import pandas as pd
import numpy as np
import os

path_macro = "/Users/camilajauregui/Documents/6to año/Tesis/datos macro/"

# ── DESEMPLEO (mensual → mantener mensual) ────────────────────────────────────
desempleo = pd.read_excel(os.path.join(path_macro, "Tasa_Desempleo.xlsx"), sheet_name="Monthly")
desempleo = desempleo.rename(columns={"observation_date": "fecha", "UNRATE": "UNRATE"})
desempleo["fecha"] = pd.to_datetime(desempleo["fecha"])
desempleo = desempleo[["fecha", "UNRATE"]]
desempleo = desempleo.sort_values("fecha").reset_index(drop=True)
desempleo["UNRATE_yoy"] = desempleo["UNRATE"].diff(12)   # variación interanual en pp (t - t-12)
print("Desempleo mensual:")
print(desempleo.head(8))

# ── TASA HIPOTECARIA (semanal → mensual) ──────────────────────────────────────
hipotecaria = pd.read_excel(os.path.join(path_macro, "Tasa_hipotecaria.xlsx"), sheet_name="Weekly, Ending Thursday")
hipotecaria = hipotecaria.rename(columns={"observation_date": "fecha", "MORTGAGE30US": "MORTGAGE30US"})
hipotecaria["fecha"] = pd.to_datetime(hipotecaria["fecha"])

hipotecaria["fecha"] = hipotecaria["fecha"].dt.to_period("M").dt.to_timestamp()
hipotecaria = hipotecaria.groupby("fecha")["MORTGAGE30US"].mean().reset_index()
print("\nHipotecaria mensual:")
print(hipotecaria.head(8))

# ── HPI (mensual → mantener mensual, growth rate mensual) ─────────────────────
hpi = pd.read_excel(os.path.join(path_macro, "hpi_monthly.xlsx"), sheet_name="HPI_PO_monthly_hist")
hpi.columns = ["fecha", "HPI"]
hpi["fecha"] = pd.to_datetime(hpi["fecha"], format="%m/%d/%y", errors="coerce")
hpi = hpi.dropna(subset=["fecha"])
hpi["HPI"] = pd.to_numeric(hpi["HPI"], errors="coerce")
hpi = hpi.dropna()
hpi["HPI_gr"] = hpi["HPI"].pct_change()
hpi = hpi[["fecha", "HPI_gr"]].dropna()
print("\nHPI mensual:")
print(hpi.head(8))

# ── PBI (ya es trimestral → solo growth rate, sin cambio) ─────────────────────
pbi = pd.read_excel(os.path.join(path_macro, "PBI.xlsx"), sheet_name="Quarterly")
pbi = pbi.rename(columns={"observation_date": "fecha", "GDPC1": "GDP"})
pbi["fecha"] = pd.to_datetime(pbi["fecha"])
pbi["trimestre"] = pbi["fecha"].dt.to_period("Q")
pbi["GDP_gr"] = pbi["GDP"].pct_change()
pbi = pbi[["trimestre", "GDP_gr"]]
print("\nPBI trimestral:")
print(pbi.head(8))

# ── PSAVERT (mensual → mantener mensual) ──────────────────────────────────────
psavert = pd.read_excel(os.path.join(path_macro, "PSAVERT.xlsx"), sheet_name="Monthly")
psavert = psavert.rename(columns={"observation_date": "fecha", "PSAVERT": "PSAVERT"})
psavert["fecha"] = pd.to_datetime(psavert["fecha"]).dt.to_period("M").dt.to_timestamp()
psavert = psavert[["fecha", "PSAVERT"]]
print("\nPSAVERT mensual:")
print(psavert.head(8))

# ── DSPIC96 (mensual → mantener mensual) ──────────────────────────────────────
dspic96 = pd.read_excel(os.path.join(path_macro, "DSPIC96.xlsx"), sheet_name="Monthly")
dspic96 = dspic96.rename(columns={"observation_date": "fecha", "DSPIC96": "DSPIC96"})
dspic96["fecha"] = pd.to_datetime(dspic96["fecha"]).dt.to_period("M").dt.to_timestamp()
dspic96 = dspic96[["fecha", "DSPIC96"]]
print("\nDSPIC96 mensual:")
print(dspic96.head(8))

# ── MDSP (trimestral) ─────────────────────────────────────────────────────────
mdsp = pd.read_excel(os.path.join(path_macro, "MDSP.xlsx"), sheet_name="Quarterly")
mdsp = mdsp.rename(columns={"observation_date": "fecha"})
mdsp["trimestre"] = pd.to_datetime(mdsp["fecha"]).dt.to_period("Q")
mdsp = mdsp[["trimestre", "MDSP"]]
print("\nMDSP trimestral:")
print(mdsp.head(8))


# ── MERGE VARIABLES MENSUALES ─────────────────────────────────────────────────
macro_mensual = desempleo.merge(hipotecaria, on="fecha", how="inner")
macro_mensual = macro_mensual.merge(hpi,      on="fecha", how="inner")
macro_mensual = macro_mensual.merge(psavert,  on="fecha", how="left")
macro_mensual = macro_mensual.merge(dspic96,  on="fecha", how="left")

# Filtrar rango 2010-01 a 2025-12
macro_mensual = macro_mensual[
    (macro_mensual["fecha"] >= "2005-01-01") &
    (macro_mensual["fecha"] <= "2025-12-31")
]
macro_mensual = macro_mensual.dropna(subset=["UNRATE", "UNRATE_yoy", "MORTGAGE30US", "HPI_gr"])

print(f"\nTabla macro mensual final: {macro_mensual.shape[0]} filas")
print(macro_mensual.to_string(index=False))

# ── FILTRAR Y MERGE VARIABLES TRIMESTRALES ────────────────────────────────────
macro_trim = pbi[
    (pbi["trimestre"] >= "2005Q1") &
    (pbi["trimestre"] <= "2025Q4")
].dropna().copy()

macro_trim = macro_trim.merge(mdsp, on="trimestre", how="left")

print(f"\nTabla macro trimestral final: {macro_trim.shape[0]} filas")
print(macro_trim.to_string(index=False))

# ── GUARDAR ────────────────────────────────────────────────────────────────────
macro_mensual.to_parquet(os.path.join(path_macro, "macro_mensual.parquet"), index=False)
print("\nMacro mensual guardada como macro_mensual.parquet")

macro_trim.to_parquet(os.path.join(path_macro, "macro_trimestral.parquet"), index=False)
print("Macro trimestral (GDP) guardada como macro_trimestral.parquet")
