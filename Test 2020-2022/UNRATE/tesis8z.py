import pandas as pd
import numpy as np
import os

path_archivos = "/Users/camilajauregui/Documents/6to año/Tesis/datos fannie/"
path_macro    = "/Users/camilajauregui/Documents/6to año/Tesis/datos macro/"

# ── PARÁMETROS ─────────────────────────────────────────────────────────────────
LGD        = 0.40
TASA_DESC  = 0.05
N_SIM      = 1_000
SEED       = 42
estados    = ["S1_current", "S2_30dpd", "S3_60dpd", "S4_default"]

np.random.seed(SEED)

# ── CARGA DE MATRICES ──────────────────────────────────────────────────────────
# Matrices PIT proyectadas (script 5)
matrices_pit = {}
for anio in [2020, 2021, 2022]:
    matrices_pit[anio] = pd.read_csv(
        os.path.join(path_archivos, f"matriz_pit_{anio}.csv"), index_col=0
    ).reindex(index=estados, columns=estados, fill_value=0)

# Matriz 2019 como comparación estática (reconstruida desde panel cohort)
panel_cohort = pd.read_parquet(os.path.join(path_archivos, "panel_cohort_anual.parquet"))

def construir_matriz(panel, anios):
    sub = panel[panel["anio"].isin(anios)]
    N   = pd.crosstab(sub["grupo"], sub["grupo_next"])
    N   = N.reindex(index=estados, columns=estados, fill_value=0)
    return N.div(N.sum(axis=1), axis=0).fillna(0)

P_2019 = construir_matriz(panel_cohort, [2019])

print("Matrices cargadas:")
print(f"  Solo 2019, Proyectadas 2020/2021/2022")

# ── CARGA DEL PANEL MENSUAL PARA EAD ──────────────────────────────────────────
panel_mensual = pd.read_parquet(os.path.join(path_archivos, "panel_mensual.parquet"))
panel_mensual["anio"] = panel_mensual["fecha"].dt.year
panel_mensual["mes"]  = panel_mensual["fecha"].dt.month

# ── FUNCIÓN: CONSTRUIR PORTFOLIO INICIAL ───────────────────────────────────────
def portfolio_inicio_anio(anio):
    enero = panel_mensual[
        (panel_mensual["anio"] == anio) &
        (panel_mensual["mes"]  == 1)
    ].copy()

    enero = (
        enero.sort_values(["LOAN_ID", "fecha"])
        .groupby("LOAN_ID")
        .first()
        .reset_index()
        [["LOAN_ID", "grupo", "CURRENT_UPB"]]
        .dropna(subset=["grupo", "CURRENT_UPB"])
    )

    enero = enero[enero["grupo"] != "S4_default"].copy()
    return enero

# ── FUNCIÓN: SIMULACIÓN MONTE CARLO ───────────────────────────────────────────
def simular_ecl(portfolio, matriz, anio, n_sim=N_SIM):
    DF = (1 + TASA_DESC) ** (-1)

    grupos  = portfolio["grupo"].values
    ead     = portfolio["CURRENT_UPB"].values
    n_loans = len(portfolio)

    cum_probs = {}
    for estado in estados:
        if estado != "S4_default":
            fila = matriz.loc[estado].values
            cum_probs[estado] = np.cumsum(fila)

    ecl_corridas = np.zeros(n_sim)

    for s in range(n_sim):
        u = np.random.uniform(size=n_loans)
        defaulteo = np.zeros(n_loans, dtype=bool)

        for idx_estado, estado in enumerate(estados[:-1]):
            mask = grupos == estado
            if not mask.any():
                continue
            cp          = cum_probs[estado]
            u_sub       = u[mask]
            idx_destino = np.searchsorted(cp, u_sub)
            defaulteo[mask] = (idx_destino == 3)

        loss = np.sum(DF * ead[defaulteo] * LGD)
        ecl_corridas[s] = loss

    return ecl_corridas

# ── FUNCIÓN: PÉRDIDA REAL OBSERVADA ───────────────────────────────────────────
def perdida_real(anio, portfolio):
    DF = (1 + TASA_DESC) ** (-1)

    cohort_anio     = panel_cohort[panel_cohort["anio"] == anio]
    loans_portfolio = set(portfolio["LOAN_ID"].values)
    defaultearon    = cohort_anio[
        (cohort_anio["LOAN_ID"].isin(loans_portfolio)) &
        (cohort_anio["grupo_next"] == "S4_default")
    ]["LOAN_ID"].values

    ead_default = portfolio[
        portfolio["LOAN_ID"].isin(defaultearon)
    ]["CURRENT_UPB"].values

    return DF * ead_default.sum() * LGD

# ══════════════════════════════════════════════════════════════════════════════
# SIMULACIÓN POR AÑO Y MATRIZ
# ══════════════════════════════════════════════════════════════════════════════

modelos = {
    "Solo 2019": P_2019,
}

resultados = []

for anio in [2020, 2021, 2022]:
    print(f"\n{'='*65}")
    print(f"AÑO {anio}")
    print("="*65)

    port = portfolio_inicio_anio(anio)
    print(f"  Préstamos activos: {len(port):,}")
    print(f"  EAD total: ${port['CURRENT_UPB'].sum():,.0f}")

    loss_real = perdida_real(anio, port)
    print(f"  Pérdida real observada: ${loss_real:,.0f}")

    # Matrices del año: solo 2019 y proyectada
    matrices_anio = {**modelos, "Proyectada": matrices_pit.get(anio, P_2019)}

    for nombre, matriz in matrices_anio.items():
        ecl_sim = simular_ecl(port, matriz, anio)

        ecl_medio  = ecl_sim.mean()
        ecl_std    = ecl_sim.std()
        ecl_p5     = np.percentile(ecl_sim, 5)
        ecl_p95    = np.percentile(ecl_sim, 95)
        error_abs  = abs(ecl_medio - loss_real)
        error_rel  = error_abs / loss_real if loss_real > 0 else np.nan
        dentro_IC  = ecl_p5 <= loss_real <= ecl_p95

        print(f"\n  [{nombre}]")
        print(f"    ECL estimado (media):  ${ecl_medio:>15,.0f}")
        print(f"    Std:                   ${ecl_std:>15,.0f}")
        print(f"    IC 90% [{ecl_p5:,.0f} — {ecl_p95:,.0f}]")
        print(f"    Pérdida real:          ${loss_real:>15,.0f}")
        print(f"    Error absoluto:        ${error_abs:>15,.0f}")
        print(f"    Error relativo:        {error_rel:>15.2%}")
        print(f"    ¿Real dentro del IC?   {'Sí ✓' if dentro_IC else 'No ✗'}")

        resultados.append({
            "anio":        anio,
            "modelo":      nombre,
            "n_loans":     len(port),
            "EAD_total":   port["CURRENT_UPB"].sum(),
            "ECL_medio":   ecl_medio,
            "ECL_std":     ecl_std,
            "ECL_p5":      ecl_p5,
            "ECL_p95":     ecl_p95,
            "loss_real":   loss_real,
            "error_abs":   error_abs,
            "error_rel":   error_rel,
            "dentro_IC":   dentro_IC
        })

# ── TABLA RESUMEN ──────────────────────────────────────────────────────────────
df_res = pd.DataFrame(resultados)

print("\n" + "="*75)
print("TABLA RESUMEN — ECL estimado vs pérdida real por año y modelo")
print("="*75)
print(f"{'Año':<6} {'Modelo':<14} {'ECL medio':>14} {'Loss real':>14} {'Error rel.':>12} {'En IC?':>8}")
print("-"*75)
for _, row in df_res.iterrows():
    print(f"{int(row['anio']):<6} {row['modelo']:<14} "
          f"${row['ECL_medio']:>13,.0f} "
          f"${row['loss_real']:>13,.0f} "
          f"{row['error_rel']:>11.2%} "
          f"{'Sí' if row['dentro_IC'] else 'No':>8}")

# ── MÉTRICAS AGREGADAS POR MODELO ─────────────────────────────────────────────
print("\n" + "="*55)
print("MÉTRICAS AGREGADAS POR MODELO (promedio 2020-2022)")
print("="*55)
print(f"{'Modelo':<14} {'MAE rel.':>12} {'% veces en IC':>15}")
print("-"*55)
for modelo in ["Solo 2019", "Proyectada"]:
    sub = df_res[df_res["modelo"] == modelo]
    mae_rel   = sub["error_rel"].mean()
    pct_ic    = sub["dentro_IC"].mean() * 100
    print(f"{modelo:<14} {mae_rel:>12.2%} {pct_ic:>14.0f}%")

# ── GUARDAR ────────────────────────────────────────────────────────────────────
df_res.to_csv(os.path.join(path_archivos, "resultados_montecarlo.csv"), index=False)
print("\nResultados guardados en resultados_montecarlo.csv")
