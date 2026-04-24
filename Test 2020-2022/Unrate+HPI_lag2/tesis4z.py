import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

path_archivos = "/Users/camilajauregui/Documents/6to año/Tesis/datos fannie/"
path_macro    = "/Users/camilajauregui/Documents/6to año/Tesis/datos macro/"
path_out      = path_archivos   # donde se guardan los PNG

# ── CARGA ──────────────────────────────────────────────────────────────────────
panel_cohort  = pd.read_parquet(os.path.join(path_archivos, "panel_cohort_anual.parquet"))
macro_mensual = pd.read_parquet(os.path.join(path_macro, "macro_mensual.parquet"))
macro_trim    = pd.read_parquet(os.path.join(path_macro, "macro_trimestral.parquet"))
panel_mensual = pd.read_parquet(os.path.join(path_archivos, "panel_mensual.parquet"))

# ── MACRO ANUAL ────────────────────────────────────────────────────────────────
# Mensuales: promedio de 12 datos por año
macro_mensual["anio"] = macro_mensual["fecha"].dt.year
macro_anual_m = (
    macro_mensual
    .groupby("anio")[["UNRATE", "MORTGAGE30US", "HPI_gr", "UNRATE_yoy", "PSAVERT", "DSPIC96"]]
    .mean()
    .reset_index()
)

# Lags de HPI_gr (1 y 2 años)
macro_anual_m = macro_anual_m.sort_values("anio").reset_index(drop=True)
macro_anual_m["HPI_gr_lag1"] = macro_anual_m["HPI_gr"].shift(1)
macro_anual_m["HPI_gr_lag2"] = macro_anual_m["HPI_gr"].shift(2)

# Trimestrales: promedio de 4 datos por año
macro_trim["anio"] = macro_trim["trimestre"].dt.year
macro_anual_g = (
    macro_trim
    .groupby("anio")[["GDP_gr", "MDSP"]]
    .mean()
    .reset_index()
)

# Merge mensual + trimestral
macro_anual = macro_anual_m.merge(macro_anual_g, on="anio", how="left")

# ── BASE PRÉSTAMO-AÑO ──────────────────────────────────────────────────────────
grupos_no_default = ["S1_current", "S2_30dpd", "S3_60dpd"]

panel_cohort["default_next"] = (panel_cohort["grupo_next"] == "S4_default").astype(int)
df_base = panel_cohort[panel_cohort["grupo"].isin(grupos_no_default)].copy()

# ── AGREGAR A NIVEL GRUPO-AÑO ─────────────────────────────────────────────────
df_group_year = (
    df_base
    .groupby(["anio", "grupo"], as_index=False)
    .agg(
        n_prestamos  = ("LOAN_ID",      "count"),
        n_defaults   = ("default_next", "sum"),
        pd_anual_obs = ("default_next", "mean"),
    )
)

df_group_year = df_group_year.merge(macro_anual, on="anio", how="left")


# Solo años train (2007-2019) para la correlación
df = df_group_year[df_group_year["anio"].between(2007, 2019)].copy()

cols_x = ["UNRATE", "UNRATE_yoy", "MORTGAGE30US",
          "HPI_gr", "HPI_gr_lag1", "HPI_gr_lag2",
          "PSAVERT", "GDP_gr", "MDSP", "DSPIC96"]
cols_y = ["pd_anual_obs"]

df = df[["anio", "grupo"] + cols_x + cols_y].dropna()

# ── TABLA DE CORRELACIONES (Pearson y Spearman) ────────────────────────────────
print("=" * 65)
print("CORRELACIONES CON pd_anual_obs  (train 2007-2019, todos los grupos)")
print("=" * 65)
print(f"{'Variable':<18} {'Pearson':>10} {'Spearman':>11}")
print("-" * 42)

pearson_vals  = {}
spearman_vals = {}
for col in cols_x:
    r_p = df[col].corr(df["pd_anual_obs"], method="pearson")
    r_s = df[col].corr(df["pd_anual_obs"], method="spearman")
    pearson_vals[col]  = r_p
    spearman_vals[col] = r_s
    print(f"  {col:<16} {r_p:>+10.4f} {r_s:>+11.4f}")

print()
print("CORRELACIONES POR GRUPO")
print("=" * 65)
for grupo in grupos_no_default:
    sub = df[df["grupo"] == grupo]
    print(f"\n  {grupo}:")
    print(f"  {'Variable':<18} {'Pearson':>10} {'Spearman':>11}")
    print("  " + "-" * 40)
    for col in cols_x:
        r_p = sub[col].corr(sub["pd_anual_obs"], method="pearson")
        r_s = sub[col].corr(sub["pd_anual_obs"], method="spearman")
        print(f"    {col:<16} {r_p:>+10.4f} {r_s:>+11.4f}")

# ── FIGURA 1: HEATMAP DE CORRELACIÓN PEARSON ──────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 4))

corr_matrix = df[cols_x + cols_y].corr(method="pearson")
mask = np.ones_like(corr_matrix, dtype=bool)
for col in cols_x:
    mask[corr_matrix.index.get_loc(col), corr_matrix.columns.get_loc("pd_anual_obs")] = False
    mask[corr_matrix.index.get_loc("pd_anual_obs"), corr_matrix.columns.get_loc(col)] = False
mask[corr_matrix.index.get_loc("pd_anual_obs"), corr_matrix.columns.get_loc("pd_anual_obs")] = False

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn",
    vmin=-1, vmax=1,
    linewidths=0.5,
    mask=mask,
    ax=ax1
)
ax1.set_title("Heatmap de correlación de Pearson — X vs pd_anual_obs", fontsize=11)
fig1.tight_layout()
fig1.savefig(os.path.join(path_out, "corr_heatmap.png"), dpi=150)
print("\nGuardado: corr_heatmap.png")

# ── FIGURA 2: SCATTER PLOTS X vs Y (coloreados por grupo) ─────────────────────
colores = {"S1_current": "#2196F3", "S2_30dpd": "#FF9800", "S3_60dpd": "#F44336"}
labels_grupo = {"S1_current": "S1 (current)", "S2_30dpd": "S2 (30 dpd)", "S3_60dpd": "S3 (60 dpd)"}

n_x = len(cols_x)
fig2, axes = plt.subplots(1, n_x, figsize=(4.5 * n_x, 4.5))

for ax, col in zip(axes, cols_x):
    for grupo in grupos_no_default:
        sub = df[df["grupo"] == grupo]
        ax.scatter(sub[col], sub["pd_anual_obs"],
                   color=colores[grupo], label=labels_grupo[grupo],
                   alpha=0.75, edgecolors="white", linewidths=0.4, s=60, zorder=3)

    # Línea de tendencia global
    x_vals = df[col].values
    y_vals = df["pd_anual_obs"].values
    coef = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_line, np.polyval(coef, x_line), color="black", linewidth=1.2,
            linestyle="--", zorder=4)

    r_p = pearson_vals[col]
    ax.set_title(f"{col}\nr = {r_p:+.3f}", fontsize=10)
    ax.set_xlabel(col, fontsize=9)
    ax.set_ylabel("pd_anual_obs" if col == cols_x[0] else "", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7)

fig2.suptitle("Scatter plots: variables X vs pd_anual_obs  (train 2007-2019)", fontsize=11, y=1.01)
fig2.tight_layout()
fig2.savefig(os.path.join(path_out, "corr_scatter.png"), dpi=150, bbox_inches="tight")
print("Guardado: corr_scatter.png")

# ── FIGURA 3: SERIES TEMPORALES X e Y (grupo S2 como referencia) ──────────────
s2 = df[df["grupo"] == "S2_30dpd"].set_index("anio").sort_index()
n_filas = (n_x + 1) // 2  # 2 columnas
fig3, axes3 = plt.subplots(n_filas, 2, figsize=(12, 4 * n_filas))
axes3_flat = axes3.flatten()

for i, col in enumerate(cols_x):
    ax_l = axes3_flat[i]
    ax_r = ax_l.twinx()

    # Serie X
    ax_l.plot(s2.index, s2[col], color="#1565C0", marker="o", markersize=4,
              linewidth=1.5, label=col)
    ax_l.set_ylabel(col, color="#1565C0", fontsize=9)
    ax_l.tick_params(axis="y", labelcolor="#1565C0")

    # Serie Y
    ax_r.plot(s2.index, s2["pd_anual_obs"], color="#B71C1C", marker="s",
              markersize=4, linewidth=1.5, linestyle="--", label="pd_anual_obs (S2)")
    ax_r.set_ylabel("pd_anual_obs (S2)", color="#B71C1C", fontsize=9)
    ax_r.tick_params(axis="y", labelcolor="#B71C1C")
    ax_r.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    r_p = pearson_vals[col]
    ax_l.set_title(f"{col}  vs  pd_anual_obs (S2)    r = {r_p:+.3f}", fontsize=9)
    ax_l.set_xlabel("Año", fontsize=8)
    ax_l.grid(True, linestyle=":", alpha=0.4)

    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_l.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7, loc="upper right")

# Ocultar subplot sobrante si n_x es impar
if n_x % 2 != 0:
    axes3_flat[-1].set_visible(False)

fig3.suptitle("Series temporales: variables X y pd_anual_obs de S2  (train 2007-2019)",
              fontsize=11, y=1.01)
fig3.tight_layout()
fig3.savefig(os.path.join(path_out, "corr_timeseries.png"), dpi=150, bbox_inches="tight")
print("Guardado: corr_timeseries.png")

plt.show()
print("\nAnálisis de correlación completado.")
