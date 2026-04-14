import pandas as pd
import numpy as np
import os

path_archivos = "/Users/camilajauregui/Documents/6to año/Tesis/datos fannie/"

# ── CARGA ──────────────────────────────────────────────────────────────────────
panel_cohort = pd.read_parquet(os.path.join(path_archivos, "panel_cohort_anual.parquet"))

estados           = ["S1_current", "S2_30dpd", "S3_60dpd", "S4_default"]
grupos_no_default = ["S1_current", "S2_30dpd", "S3_60dpd"]

# Matrices PIT proyectadas (del script 5)
matrices_pit = {}
for anio in [2023, 2024, 2025]:
    matrices_pit[anio] = pd.read_csv(
        os.path.join(path_archivos, f"matriz_pit_{anio}.csv"), index_col=0
    ).reindex(index=estados, columns=estados, fill_value=0)

# ── CONSTRUIR MATRICES BASE ────────────────────────────────────────────────────

def construir_matriz(panel, anios):
    sub = panel[panel["anio"].isin(anios)]
    N   = pd.crosstab(sub["grupo"], sub["grupo_next"])
    N   = N.reindex(index=estados, columns=estados, fill_value=0)
    return N.div(N.sum(axis=1), axis=0).fillna(0)

# Matriz 2022 (comparación estática — último año de train)
P_2022 = construir_matriz(panel_cohort, [2022])

print("Matriz 2022 (comparación estática):")
print(P_2022.round(4).to_string())

# ── PD OBSERVADA POR AÑO Y GRUPO (test) ───────────────────────────────────────
pd_obs  = {}
mat_obs = {}
for anio in [2023, 2024, 2025]:
    sub   = panel_cohort[panel_cohort["anio"] == anio]
    N_obs = pd.crosstab(sub["grupo"], sub["grupo_next"])
    N_obs = N_obs.reindex(index=estados, columns=estados, fill_value=0)
    P_obs = N_obs.div(N_obs.sum(axis=1), axis=0).fillna(0)
    pd_obs[anio]  = P_obs["S4_default"]
    mat_obs[anio] = P_obs

print("\nMATRICES OBSERVADAS POR AÑO (test):")
for anio in [2023, 2024, 2025]:
    print(f"\n{'='*60}")
    print(f"MATRIZ OBSERVADA {anio}")
    print("="*60)
    print(mat_obs[anio].round(4).to_string())

# ── FUNCIÓN: POTENCIA DE MATRIZ ────────────────────────────────────────────────
def potencia_matriz(P, n):
    """Eleva la matriz P a la potencia n mediante multiplicación matricial."""
    resultado = np.eye(len(P))
    for _ in range(n):
        resultado = resultado @ P.values
    return pd.DataFrame(resultado, index=P.index, columns=P.columns)

# ── COMPARACIÓN AÑO A AÑO ─────────────────────────────────────────────────────
# Para predecir 1 año adelante (2023): P^1
# Para predecir 2 años adelante (2024): P^2
# Para predecir 3 años adelante (2025): P^3

print("\n" + "="*70)
print("COMPARACIÓN AÑO A AÑO")
print("="*70)

resultados_anuales = []

for anio in [2023, 2024, 2025]:
    pd_2022_h = P_2022["S4_default"]
    pd_proy_h = matrices_pit[anio]["S4_default"]

    print(f"\n{'─'*95}")
    print(f"VALIDACION {anio}")
    print(f"{'─'*95}")
    print(f"{'Grupo':<18} {'Observada':>10} {'Proyectada':>12} {'MAE':>8} {'Solo 2022':>12} {'MAE':>8}")
    print("-"*75)

    for g in grupos_no_default:
        obs   = pd_obs[anio][g]
        proy  = pd_proy_h[g]
        s2022 = pd_2022_h[g]
        print(f"{g:<18} {obs:>10.6f} {proy:>12.6f} {abs(proy-obs):>8.6f} {s2022:>12.6f} {abs(s2022-obs):>8.6f}")
        resultados_anuales.append({
            "anio":       anio,
            "grupo":      g,
            "observada":  obs,
            "proyectada": proy,
            "solo_2022":  s2022
        })

# ── MÉTRICAS POR MODELO Y HORIZONTE ───────────────────────────────────────────
df_res = pd.DataFrame(resultados_anuales)

print("\n" + "="*70)
print("MÉTRICAS POR HORIZONTE (promedio sobre grupos no default)")
print("="*70)
print(f"{'Año':<12} {'Modelo':<14} {'MAE':>10} {'RMSE':>10} {'Error rel.':>12}")
print("-"*70)

modelos_cols = {"Proyectada": "proyectada", "Solo 2022": "solo_2022"}

metricas = []
for anio_m in [2023, 2024, 2025]:
    sub = df_res[df_res["anio"] == anio_m]
    for nombre, col in modelos_cols.items():
        errores   = sub[col] - sub["observada"]
        mae       = errores.abs().mean()
        rmse      = np.sqrt((errores**2).mean())
        err_rel   = (errores.abs() / sub["observada"].replace(0, np.nan)).mean()
        print(f"{anio_m:<12} {nombre:<14} {mae:>10.6f} {rmse:>10.6f} {err_rel:>12.4f}")
        metricas.append({
            "anio":      anio_m,
            "modelo":    nombre,
            "MAE":       mae,
            "RMSE":      rmse,
            "error_rel": err_rel
        })

# ── MATRICES Y PD A 3 AÑOS ────────────────────────────────────────────────────
print("\n" + "="*70)
print("PD A 3 AÑOS POR GRUPO — los tres modelos")
print("="*70)

# Proyectada: P_2023 × P_2024 × P_2025
P_proy_3 = pd.DataFrame(
    matrices_pit[2023].values @ matrices_pit[2024].values @ matrices_pit[2025].values,
    index=estados, columns=estados
)

# Solo 2022: P_2022^3
P_2022_3 = potencia_matriz(P_2022, 3)

# PD observada a 3 años: M_obs_2023 x M_obs_2024 x M_obs_2025
M_obs_3 = pd.DataFrame(
    mat_obs[2023].values @ mat_obs[2024].values @ mat_obs[2025].values,
    index=estados, columns=estados
)
pd_obs_3 = M_obs_3["S4_default"].to_dict()

print(f"\n{'Grupo':<18} {'Observada':>10} {'Proyectada':>12} {'Solo 2022':>12}")
print("-"*60)

resultados_3 = []
for g in grupos_no_default:
    obs   = pd_obs_3[g]
    proy  = P_proy_3.loc[g, "S4_default"]
    s2022 = P_2022_3.loc[g, "S4_default"]
    print(f"{g:<18} {obs:>10.6f} {proy:>12.6f} {s2022:>12.6f}")
    resultados_3.append({
        "grupo":      g,
        "observada":  obs,
        "proyectada": proy,
        "solo_2022":  s2022
    })

df_3 = pd.DataFrame(resultados_3)

# Métricas a 3 años
print(f"\n{'Modelo':<14} {'MAE':>10} {'RMSE':>10} {'Error rel.':>12}")
print("-"*50)
for nombre, col in modelos_cols.items():
    errores = df_3[col] - df_3["observada"]
    mae     = errores.abs().mean()
    rmse    = np.sqrt((errores**2).mean())
    err_rel = (errores.abs() / df_3["observada"].replace(0, np.nan)).mean()
    print(f"{nombre:<14} {mae:>10.6f} {rmse:>10.6f} {err_rel:>12.4f}")

# ── GUARDAR ────────────────────────────────────────────────────────────────────
df_res.to_csv(os.path.join(path_archivos, "comparacion_anual.csv"), index=False)
df_3.to_csv(os.path.join(path_archivos, "comparacion_3anios.csv"), index=False)
pd.DataFrame(metricas).to_csv(os.path.join(path_archivos, "metricas_modelos.csv"), index=False)

P_proy_3.to_csv(os.path.join(path_archivos, "matriz_largo_plazo_proyectada.csv"))
P_2022_3.to_csv(os.path.join(path_archivos, "matriz_largo_plazo_2022.csv"))

print("\nResultados y matrices guardados.")