import pandas as pd
import numpy as np
import os
import glob
import zipfile

# ── COLUMNAS ───────────────────────────────────────────────────────────────────
todas_las_columnas = [
    "POOL_ID", "LOAN_ID", "ACT_PERIOD", "CHANNEL", "SELLER", "SERVICER",
    "MASTER_SERVICER", "ORIG_RATE", "CURR_RATE", "ORIG_UPB", "ISSUANCE_UPB",
    "CURRENT_UPB", "ORIG_TERM", "ORIG_DATE", "FIRST_PAY", "LOAN_AGE",
    "REM_MONTHS", "ADJ_REM_MONTHS", "MATR_DT", "OLTV", "OCLTV",
    "NUM_BO", "DTI", "CSCORE_B", "CSCORE_C", "FIRST_FLAG", "PURPOSE",
    "PROP", "NO_UNITS", "OCC_STAT", "STATE", "MSA", "ZIP", "MI_PCT",
    "PRODUCT", "PPMT_FLG", "IO", "FIRST_PAY_IO", "MNTHS_TO_AMTZ_IO",
    "DLQ_STATUS", "PMT_HISTORY", "MOD_FLAG", "MI_CANCEL_FLAG", "Zero_Bal_Code",
    "ZB_DTE", "LAST_UPB", "RPRCH_DTE", "CURR_SCHD_PRNCPL", "TOT_SCHD_PRNCPL",
    "UNSCHD_PRNCPL_CURR", "LAST_PAID_INSTALLMENT_DATE", "FORECLOSURE_DATE",
    "DISPOSITION_DATE", "FORECLOSURE_COSTS", "PROPERTY_PRESERVATION_AND_REPAIR_COSTS",
    "ASSET_RECOVERY_COSTS", "MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS",
    "ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY", "NET_SALES_PROCEEDS",
    "CREDIT_ENHANCEMENT_PROCEEDS", "REPURCHASES_MAKE_WHOLE_PROCEEDS",
    "OTHER_FORECLOSURE_PROCEEDS", "NON_INTEREST_BEARING_UPB",
    "PRINCIPAL_FORGIVENESS_AMOUNT", "ORIGINAL_LIST_START_DATE",
    "ORIGINAL_LIST_PRICE", "CURRENT_LIST_START_DATE", "CURRENT_LIST_PRICE",
    "ISSUE_SCOREB", "ISSUE_SCOREC", "CURR_SCOREB", "CURR_SCOREC",
    "MI_TYPE", "SERV_IND", "CURRENT_PERIOD_MODIFICATION_LOSS_AMOUNT",
    "CUMULATIVE_MODIFICATION_LOSS_AMOUNT", "CURRENT_PERIOD_CREDIT_EVENT_NET_GAIN_OR_LOSS",
    "CUMULATIVE_CREDIT_EVENT_NET_GAIN_OR_LOSS", "HOMEREADY_PROGRAM_INDICATOR",
    "FORECLOSURE_PRINCIPAL_WRITE_OFF_AMOUNT", "RELOCATION_MORTGAGE_INDICATOR",
    "ZERO_BALANCE_CODE_CHANGE_DATE", "LOAN_HOLDBACK_INDICATOR",
    "LOAN_HOLDBACK_EFFECTIVE_DATE", "DELINQUENT_ACCRUED_INTEREST",
    "PROPERTY_INSPECTION_WAIVER_INDICATOR", "HIGH_BALANCE_LOAN_INDICATOR",
    "ARM_5_YR_INDICATOR", "ARM_PRODUCT_TYPE", "MONTHS_UNTIL_FIRST_PAYMENT_RESET",
    "MONTHS_BETWEEN_SUBSEQUENT_PAYMENT_RESET", "INTEREST_RATE_CHANGE_DATE",
    "PAYMENT_CHANGE_DATE", "ARM_INDEX", "ARM_CAP_STRUCTURE",
    "INITIAL_INTEREST_RATE_CAP", "PERIODIC_INTEREST_RATE_CAP",
    "LIFETIME_INTEREST_RATE_CAP", "MARGIN", "BALLOON_INDICATOR",
    "PLAN_NUMBER", "FORBEARANCE_INDICATOR", "HIGH_LOAN_TO_VALUE_HLTV_REFINANCE_OPTION_INDICATOR",
    "DEAL_NAME", "RE_PROCS_FLAG", "ADR_TYPE", "ADR_COUNT", "ADR_UPB",
    "PAYMENT_DEFERRAL_MOD_EVENT_FLAG", "INTEREST_BEARING_UPB"
]

columnas_necesarias = [
    "LOAN_ID", "ACT_PERIOD", "CURR_RATE", "ORIG_UPB",
    "CURRENT_UPB", "ORIG_DATE", "CSCORE_B",
    "DLQ_STATUS", "Zero_Bal_Code", "ZB_DTE"
]

path_archivos = "/Users/camilajauregui/Documents/6to año/Tesis/datos fannie/"

# ── CARGA ──────────────────────────────────────────────────────────────────────
lista_dfs = []

for archivo_zip in sorted(glob.glob(os.path.join(path_archivos, "*.zip"))):
    print(f"Procesando {archivo_zip}...")
    with zipfile.ZipFile(archivo_zip, 'r') as z:
        nombre_csv = z.namelist()[0]
        z.extract(nombre_csv, path_archivos)
    ruta_csv = os.path.join(path_archivos, nombre_csv)
    df_temp = pd.read_csv(
        ruta_csv, sep="|", header=None,
        names=todas_las_columnas, usecols=columnas_necesarias,
        dtype=str, low_memory=False, nrows=1_000_000
    )
    lista_dfs.append(df_temp)
    print(f"  → {df_temp.shape[0]:,} filas")
    os.remove(ruta_csv)
    print(f"  → CSV borrado")

panel = pd.concat(lista_dfs, ignore_index=True)
print(f"\nPanel completo: {panel.shape[0]:,} filas")

# ── LIMPIEZA DE TIPOS ──────────────────────────────────────────────────────────
panel["ACT_PERIOD"] = panel["ACT_PERIOD"].astype(str).str.strip().str.zfill(6)
panel["fecha"] = pd.to_datetime(panel["ACT_PERIOD"], format="%m%Y", errors="coerce")

panel["DLQ_STATUS"]    = pd.to_numeric(panel["DLQ_STATUS"],    errors="coerce")
panel["CSCORE_B"]      = pd.to_numeric(panel["CSCORE_B"],      errors="coerce")
panel["ORIG_UPB"]      = pd.to_numeric(panel["ORIG_UPB"],      errors="coerce")
panel["CURRENT_UPB"]   = pd.to_numeric(panel["CURRENT_UPB"],   errors="coerce")
panel["CURR_RATE"]     = pd.to_numeric(panel["CURR_RATE"],     errors="coerce")
panel["Zero_Bal_Code"] = panel["Zero_Bal_Code"].astype(str).str.strip()

# Eliminar filas sin identificador o sin fecha válida
panel = panel.dropna(subset=["LOAN_ID", "fecha"]).copy()

# Filtrar saldo cero
panel = panel[panel["CURRENT_UPB"] > 0].copy()
print(f"Filas activas (saldo > 0): {panel.shape[0]:,}")

# ── DEFAULT ────────────────────────────────────────────────────────────────────
codigos_default = ["03", "09", "97"]
panel["default"] = (
    (panel["DLQ_STATUS"] >= 3) |
    (panel["Zero_Bal_Code"].isin(codigos_default))
).astype(int)

# ── GRUPOS DE RATING ───────────────────────────────────────────────────────────
def asignar_grupo(dlq, default_flag):
    if default_flag == 1:
        return "S4_default"
    if pd.isna(dlq):
        return None
    if dlq == 0:
        return "S1_current"
    if dlq == 1:
        return "S2_30dpd"
    if dlq == 2:
        return "S3_60dpd"
    return None

panel["grupo"] = panel.apply(
    lambda x: asignar_grupo(x["DLQ_STATUS"], x["default"]), axis=1
)

# ── ORDENAR POR FECHA ──────────────────────────────────────────────────────────
panel = panel.sort_values(["LOAN_ID", "fecha"]).copy()

# ── HACER S4 ABSORBENTE ────────────────────────────────────────────────────────
panel["default_acum"] = panel.groupby("LOAN_ID")["default"].cummax()
panel.loc[panel["default_acum"] == 1, "default"] = 1
panel.loc[panel["default_acum"] == 1, "grupo"]   = "S4_default"
panel = panel.drop(columns=["default_acum"])

print(f"\nDistribución de grupos (panel mensual completo):")
print(panel["grupo"].value_counts(dropna=False))

# ── GUARDAR PANEL MENSUAL ──────────────────────────────────────────────────────
panel.to_parquet(
    os.path.join(path_archivos, "panel_mensual.parquet"), index=False
)
print("\nPanel mensual guardado como panel_mensual.parquet")
