import pandas as pd
import numpy as np
import os
import pickle

path_archivos = "/Users/camilajauregui/Documents/6to año/Tesis/datos fannie/"

# ── CARGA ──────────────────────────────────────────────────────────────────────
# Matriz base: último año de train (2019) — base para rolling de matrices PIT
panel_cohort = pd.read_parquet(os.path.join(path_archivos, "panel_cohort_anual.parquet"))

estados = ["S1_current", "S2_30dpd", "S3_60dpd", "S4_default"]

sub_2019 = panel_cohort[panel_cohort["anio"] == 2019]
N_2019   = pd.crosstab(sub_2019["grupo"], sub_2019["grupo_next"])
N_2019   = N_2019.reindex(index=estados, columns=estados, fill_value=0)
P_2019   = N_2019.div(N_2019.sum(axis=1), axis=0).fillna(0)

print("Matriz base 2019 (último año de train):")
print(P_2019.round(4).to_string())

# PD predicha por grupo y año: viene del script 4
pd_pred_df = pd.read_csv(os.path.join(path_archivos, "pd_anual_pred.csv"), index_col=0)
pd_pred_df.columns = pd_pred_df.columns.astype(int)

print("\nPD predicha por grupo y año (del script 4):")
print(pd_pred_df.round(6).to_string())

# ── PDinc Y MATRICES PIT (acumulativo) ────────────────────────────────────────
# Para cada año:
#   PDinc = PD_predicha_año - PD_S4 de la matriz BASE de ese año
#   2020: base = P_2019
#   2021: base = PIT 2020
#   2022: base = PIT 2021
# Así tanto el PDinc como la descomposición son acumulativos.

grupos_no_default = ["S1_current", "S2_30dpd", "S3_60dpd"]

# ── CONSTRUCCIÓN DE MATRICES PIT ──────────────────────────────────────────────
# Rescalado proporcional:
#   - La columna S4_default de cada fila toma directamente el valor PD_predicha
#   - El resto (1 - PD_predicha) se distribuye entre las otras columnas
#     en proporción a sus pesos actuales en la matriz base

def descomponer_matriz(P_base, pd_pred_vector, estados):
    P_new = P_base.copy()

    cols_no_default = [e for e in estados if e != "S4_default"]

    for estado in estados:
        if estado == "S4_default":
            continue
        if estado not in pd_pred_vector:
            continue

        pd_pred = pd_pred_vector[estado]

        # Peso actual de las columnas no-default en la fila base
        pesos = P_base.loc[estado, cols_no_default]
        total_pesos = pesos.sum()

        # Asignar PD predicha directo a default
        P_new.loc[estado, "S4_default"] = pd_pred

        # Distribuir el resto proporcionalmente
        if total_pesos > 0:
            P_new.loc[estado, cols_no_default] = pesos / total_pesos * (1 - pd_pred)
        else:
            # Si no hay peso previo, distribuir equitativamente
            P_new.loc[estado, cols_no_default] = (1 - pd_pred) / len(cols_no_default)

    return P_new


print("\n" + "="*60)
print("PDinc Y MATRICES PIT (acumulativo)")
print("="*60)

matrices_pit = {}
P_base_anio  = P_2019.copy()

for anio in [2020, 2021, 2022]:
    pd_pred_anio = {}

    print(f"\n  {'Grupo':<18} {'PD_pred':>10} {'PD_base':>10}")
    print(f"  {'-'*40}")
    for grupo in grupos_no_default:
        if grupo not in pd_pred_df.index:
            continue
        pd_pred = pd_pred_df.loc[grupo, anio]
        pd_ini  = P_base_anio.loc[grupo, "S4_default"]
        pd_pred_anio[grupo] = pd_pred
        print(f"  {grupo:<18} {pd_pred:>10.6f} {pd_ini:>10.6f}")

    P_pit = descomponer_matriz(P_base_anio, pd_pred_anio, estados)
    matrices_pit[anio] = P_pit
    P_base_anio = P_pit.copy()

    print(f"\n{'='*60}")
    print(f"MATRIZ PIT {anio}")
    print("="*60)
    print(P_pit.round(4).to_string())

    sumas = P_pit.sum(axis=1).round(6)
    print(f"\nSuma de filas: {sumas.to_dict()}")


# ── GUARDAR MATRICES PIT ──────────────────────────────────────────────────────
for anio in [2020, 2021, 2022]:
    matrices_pit[anio].to_csv(
        os.path.join(path_archivos, f"matriz_pit_{anio}.csv")
    )

print("\nMatrices PIT 2020, 2021 y 2022 guardadas.")