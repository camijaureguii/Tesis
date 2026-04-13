import pandas as pd
import numpy as np
import os
import pickle
import warnings
import statsmodels.api as sm

warnings.filterwarnings("ignore")

path_archivos = "/Users/camilajauregui/Documents/6to año/Tesis/datos fannie/"
path_macro    = "/Users/camilajauregui/Documents/6to año/Tesis/datos macro/"

# ── CARGA ──────────────────────────────────────────────────────────────────────
panel_cohort  = pd.read_parquet(os.path.join(path_archivos, "panel_cohort_anual.parquet"))
macro_mensual = pd.read_parquet(os.path.join(path_macro, "macro_mensual.parquet"))
macro_trim    = pd.read_parquet(os.path.join(path_macro, "macro_trimestral.parquet"))

print(f"Panel cohort anual: {panel_cohort.shape[0]:,} filas")
print(f"Macro mensual: {macro_mensual.shape[0]:,} filas")
print(f"Macro trimestral: {macro_trim.shape[0]:,} filas")

estados           = ["S1_current", "S2_30dpd", "S3_60dpd", "S4_default"]
grupos_no_default = ["S1_current", "S2_30dpd", "S3_60dpd"]

# ── MACRO ANUAL ────────────────────────────────────────────────────────────────
macro_mensual["anio"] = macro_mensual["fecha"].dt.year
macro_anual_m = (
    macro_mensual
    .groupby("anio")[["UNRATE", "UNRATE_yoy", "MORTGAGE30US", "HPI_gr", "PSAVERT", "DSPIC96"]]
    .mean()
    .reset_index()
)

macro_trim["anio"] = macro_trim["trimestre"].dt.year
macro_anual_g = (
    macro_trim
    .groupby("anio")[["GDP_gr", "MDSP"]]
    .mean()
    .reset_index()
)

macro_anual = macro_anual_m.merge(macro_anual_g, on="anio", how="left")

# Lag 2 años de HPI_gr
macro_anual = macro_anual.sort_values("anio").reset_index(drop=True)
macro_anual["HPI_gr_lag2"] = macro_anual["HPI_gr"].shift(2)

print("\nMacro anual:")
print(macro_anual[["anio", "UNRATE", "HPI_gr", "HPI_gr_lag2"]].to_string(index=False))

# ── CONSTRUIR BASE A NIVEL PRÉSTAMO ───────────────────────────────────────────
panel_cohort["default_next"] = (panel_cohort["grupo_next"] == "S4_default").astype(int)

df_base = panel_cohort[panel_cohort["grupo"].isin(grupos_no_default)].copy()

# ── AGREGAR A NIVEL GRUPO-AÑO ─────────────────────────────────────────────────
df_group_year = (
    df_base
    .groupby(["anio", "grupo"], as_index=False)
    .agg(
        n_prestamos  = ("LOAN_ID",       "count"),
        n_defaults   = ("default_next",  "sum"),
        pd_anual_obs = ("default_next",  "mean"),
    )
)

df_group_year = df_group_year.merge(macro_anual, on="anio", how="left")

print(f"\nTabla grupo-año: {df_group_year.shape[0]} filas")
print(df_group_year[["anio", "grupo", "n_prestamos", "n_defaults", "pd_anual_obs"]].to_string(index=False))

# ── 3 MODELOS GLM BINOMIAL SEPARADOS ──────────────────────────────────────────
# S1/S2: UNRATE + PSAVERT | S3: UNRATE solo (PSAVERT p=0.844, no significativo)
# features_full: para mostrar p-valor en coeficientes (incluye vars no significativas)
# features_por_grupo: para predicciones (solo vars significativas)
features_full = {
    "S1_current": ["UNRATE", "PSAVERT"],
    "S2_30dpd":   ["UNRATE", "PSAVERT"],
    "S3_60dpd":   ["UNRATE", "PSAVERT"],
}
features_por_grupo = {
    "S1_current": ["UNRATE", "PSAVERT"],
    "S2_30dpd":   ["UNRATE", "PSAVERT"],
    "S3_60dpd":   ["UNRATE"],
}
anios_test  = [2020, 2021, 2022]
modelos     = {}
pd_anual_pred = {}

for grupo in grupos_no_default:
    features        = features_por_grupo[grupo]
    features_full_g = features_full[grupo]
    df_g = df_group_year[df_group_year["grupo"] == grupo].copy()

    train_g = df_g[df_g["anio"].between(2007, 2019)].dropna(subset=features_full_g + ["pd_anual_obs"]).copy()
    test_g  = df_g[df_g["anio"].between(2020, 2022)].dropna(subset=features).copy()

    print(f"\nTrain {grupo}: {train_g.shape[0]} obs | Test: {test_g.shape[0]} obs")

    y_train = np.column_stack([
        train_g["n_defaults"],
        train_g["n_prestamos"] - train_g["n_defaults"]
    ])

    # Modelo completo (para mostrar p-valores en coeficientes)
    X_train_full = sm.add_constant(train_g[features_full_g].copy())
    modelo_full  = sm.GLM(y_train, X_train_full, family=sm.families.Binomial()).fit()
    print("\n" + "="*60)
    print(f"MODELO GLM COMPLETO (coeficientes) — {grupo}")
    print("="*60)
    print(modelo_full.summary())

    # Modelo restringido (para predicciones)
    X_train  = sm.add_constant(train_g[features].copy())
    modelo_g = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
    modelos[grupo] = modelo_g

    if features != features_full_g:
        print(f"\n  → Modelo restringido para predicciones ({grupo}): {features}")
        print(modelo_g.summary())

    train_g["pd_fitted"] = modelo_g.fittedvalues
    print(f"\nPD OBSERVADA vs AJUSTADA EN TRAIN — {grupo}")
    print("-"*45)
    print(train_g[["anio", "pd_anual_obs", "pd_fitted"]].round(6).to_string(index=False))

    X_test = sm.add_constant(test_g[features].copy(), has_constant="add")
    X_test = X_test.reindex(columns=X_train.columns, fill_value=1.0)
    test_g["pd_pred"] = modelo_g.predict(X_test)

    for _, row in test_g.iterrows():
        anio = row["anio"]
        if anio not in pd_anual_pred:
            pd_anual_pred[anio] = {}
        pd_anual_pred[anio][grupo] = row["pd_pred"]

# ── RESUMEN PREDICCIONES TEST ──────────────────────────────────────────────────
print("\n" + "="*60)
print("PD PREDICHA POR GRUPO Y AÑO (test 2023-2025)")
print("="*60)
for anio in anios_test:
    print(f"\n{anio}:")
    for grupo, pd_val in pd_anual_pred.get(anio, {}).items():
        print(f"  {grupo}: {pd_val:.6f}")

# ── GUARDAR ────────────────────────────────────────────────────────────────────
with open(os.path.join(path_archivos, "modelo_glm_anual.pkl"), "wb") as f:
    pickle.dump(modelos, f)

pd.DataFrame(pd_anual_pred).to_csv(
    os.path.join(path_archivos, "pd_anual_pred.csv")
)

macro_anual[macro_anual["anio"].isin(anios_test)].to_parquet(
    os.path.join(path_archivos, "macro_obs_2020_2022.parquet"), index=False
)

print("\nModelos GLM (3 grupos) y PD predichas guardados.")
