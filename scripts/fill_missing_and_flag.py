import pandas as pd
import openpyxl

# Paths
BASE_PATH    = "../data/processed/Base maestra.xlsx"
NULOS_PATH   = "../../Nulos_faltantes.xlsx"
OUTPUT_PATH  = "../data/processed/Base maestra.xlsx"

# Column indices (positional, to avoid encoding issues)
COL_POL = 31   # Pol en caña en bascula o coresampler
COL_PUR = 32   # Pureza jugo Core sampler
COL_REC = 33   # Recuperación Total

# Load base
df = pd.read_excel(BASE_PATH)
col_pol = df.columns[COL_POL]
col_pur = df.columns[COL_PUR]
col_rec = df.columns[COL_REC]

print(f"Base maestra loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# ── STEP 1: Fill missing values from Nulos_faltantes ──────────────────────────
# Source file contains manually collected values for Nov 2024-2025.
# Read with openpyxl (data_only=True) to get real values, not cached formulas.

wb = openpyxl.load_workbook(NULOS_PATH, data_only=True)
ws = wb.active

nulos = {}
for row in ws.iter_rows(min_row=3, values_only=True):  # row 1 empty, row 2 header
    ingenio = row[1]
    if ingenio:
        nulos[ingenio] = {"pol": row[4], "pur": row[5], "rec": row[6]}

# Name mapping: Nulos_faltantes uses uppercase, Base maestra uses title case
name_map = {
    "PANTALEON":   "Pantaleon",
    "PALO GORDO":  "Palo Gordo",
    "MADRE TIERRA":"Madre Tierra",
    "LA UNION":    "La Uni\u00f3n",
    "SANTA ANA":   "Santa Ana",
    "TRINIDAD":    "Trinidad",
}

filled = 0
mask_nov = (df["Zafra"] == "2024-2025") & (df["Mes"] == "Noviembre")

for idx, row in df[mask_nov].iterrows():
    key = name_map.get(row["Ingenio"].upper())
    if key is None:
        key = row["Ingenio"]
    nulos_key = next((k for k in nulos if name_map.get(k) == row["Ingenio"]), None)
    if nulos_key is None:
        continue
    vals = nulos[nulos_key]
    if pd.isna(row[col_pol]):
        df.at[idx, col_pol] = vals["pol"]
        filled += 1
    if pd.isna(row[col_pur]):
        df.at[idx, col_pur] = vals["pur"]
    if pd.isna(row[col_rec]):
        df.at[idx, col_rec] = vals["rec"]

print(f"Step 1: {filled} rows filled from Nulos_faltantes (Nov 2024-2025)")

# ── STEP 2: Flag — ausencia_viscosidad ────────────────────────────────────────
# Viscosidad_25C and Viscosidad_40C are missing for all mills in 2020-2021.
# The viscometer was not available during that first season.

df["ausencia_viscosidad"] = None
df.loc[df["Zafra"] == "2020-2021", "ausencia_viscosidad"] = "sin_equipo"

print(f"Step 2: ausencia_viscosidad — {(df['ausencia_viscosidad'] == 'sin_equipo').sum()} rows flagged")

# ── STEP 3: Flag — ausencia_coresampler ───────────────────────────────────────
# Covers: Pol en caña en bascula o coresampler + Pureza jugo Core sampler
#
# Tulula (all seasons): mill never had a coresampler, so these variables
#   were never measured.
#
# Magdalena (Nov and May, seasons 2023-2024 and 2024-2025): coresampler data
#   is absent because those months fall outside their harvest window
#   (season starts late in Nov, ends early before May).
#
# Madre Tierra (May 2024-2025): season ended earlier than expected that year.

df["ausencia_coresampler"] = None

df.loc[df["Ingenio"] == "Tulula", "ausencia_coresampler"] = "sin_equipo"

mask_mag_border = (
    (df["Ingenio"] == "Magdalena") &
    (df["Zafra"].isin(["2023-2024", "2024-2025"])) &
    (df["Mes"].str.strip().isin(["Noviembre", "Mayo"]))
)
df.loc[mask_mag_border, "ausencia_coresampler"] = "fuera_zafra"

mask_mt_may = (
    (df["Ingenio"] == "Madre Tierra") &
    (df["Zafra"] == "2024-2025") &
    (df["Mes"].str.strip() == "Mayo")
)
df.loc[mask_mt_may, "ausencia_coresampler"] = "fuera_zafra"

n_seq = (df["ausencia_coresampler"] == "sin_equipo").sum()
n_fz  = (df["ausencia_coresampler"] == "fuera_zafra").sum()
print(f"Step 3: ausencia_coresampler — {n_seq} sin_equipo, {n_fz} fuera_zafra")

# ── STEP 4: Flag — ausencia_recuperacion ──────────────────────────────────────
# Covers: Recuperación Total
#
# Magdalena (all seasons): this variable was never collected for this mill.
#   All records are NaN across the entire dataset.
#
# Madre Tierra (May 2024-2025): season ended early, no production that month.
#
# Tulula (Feb 2021-2022 and Mar 2024-2025): isolated months with no recorded
#   value, likely not reported that period.

df["ausencia_recuperacion"] = None

df.loc[df["Ingenio"] == "Magdalena", "ausencia_recuperacion"] = "no_registrado"

df.loc[mask_mt_may, "ausencia_recuperacion"] = "fuera_zafra"

mask_tul_gaps = (
    (df["Ingenio"] == "Tulula") &
    (
        ((df["Zafra"] == "2021-2022") & (df["Mes"].str.strip() == "Febrero")) |
        ((df["Zafra"] == "2024-2025") & (df["Mes"].str.strip() == "Marzo"))
    )
)
df.loc[mask_tul_gaps, "ausencia_recuperacion"] = "no_registrado"

n_nr = (df["ausencia_recuperacion"] == "no_registrado").sum()
n_fz = (df["ausencia_recuperacion"] == "fuera_zafra").sum()
print(f"Step 4: ausencia_recuperacion — {n_nr} no_registrado, {n_fz} fuera_zafra")

# ── STEP 5: Flag — ausencia_balance_mf ────────────────────────────────────────
# Covers: kg/t_MF, PerdPol_MF, PerdSac_MF, Perd_Indet
#
# Madre Tierra (May 2024-2025): values were not recorded for this month,
#   likely because the season had already ended.

df["ausencia_balance_mf"] = None
df.loc[mask_mt_may, "ausencia_balance_mf"] = "no_registrado"

print(f"Step 5: ausencia_balance_mf — {(df['ausencia_balance_mf'] == 'no_registrado').sum()} rows flagged")

# ── SAVE ──────────────────────────────────────────────────────────────────────
df.to_excel(OUTPUT_PATH, index=False)
print(f"\nFile saved: {OUTPUT_PATH}")
print(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")

# Quick check — no unflagged NaN should remain in key columns
flagged_cols = [col_pol, col_pur, col_rec,
                "Viscosidad_25C", "Viscosidad_40C",
                "kg/t_MF", "PerdPol_MF", "PerdSac_MF", "Perd_Indet"]
flag_cols    = ["ausencia_viscosidad", "ausencia_coresampler",
                "ausencia_recuperacion", "ausencia_balance_mf"]
other_cols   = [c for c in df.columns if c not in flagged_cols + flag_cols]
remaining    = df[other_cols].isnull().sum().sum()
print(f"Unflagged NaN in other columns: {remaining}")
