import pandas as pd

# File paths
INPUT_PATH = "../data/processed/Base maestra.xlsx"
OUTPUT_PATH = "../data/processed/Base maestra.xlsx"

# Load data
df = pd.read_excel(INPUT_PATH)

print(f"Data loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# ── CLEANING STEPS ────────────────────────────────────────────────────────────

# STEP 1: Clear Tulula values in pol and pureza columns (non-homogeneous method)
col_pol = df.columns[31]
col_pur = df.columns[32]

mask_tulula = df['Ingenio'] == 'Tulula'
df.loc[mask_tulula, col_pol] = None
df.loc[mask_tulula, col_pur] = None

print(f"Tulula: cleared values in '{col_pol}' and '{col_pur}'")

# STEP 2: Clear all Magdalena values in pol, pureza and recovery columns
col_rec = df.columns[33]

mask_magdalena = df['Ingenio'] == 'Magdalena'
df.loc[mask_magdalena, col_pol] = None
df.loc[mask_magdalena, col_pur] = None
df.loc[mask_magdalena, col_rec] = None

print(f"Magdalena: cleared values in '{col_pol}', '{col_pur}' and '{col_rec}'")

# STEP 3: Fill Magdalena (seasons 2020-2021 to 2022-2023) from Mensual_datos sheet
ceng = pd.read_excel("../../Magdalena_datos Cengicaña.xlsx", sheet_name="Mensual_datos", header=1)
ceng.columns = ceng.columns.str.strip()
ceng = ceng.rename(columns={"Mensualidad": "Mes", "Pol (kg/t)": "_pol", "Pureza jugo": "_pur"})
ceng["Zafra"] = ceng["Zafra"].str.strip()

merged = 0
for _, row in ceng[ceng["Ingenio"] == "Magdalena"].iterrows():
    mask = (df["Ingenio"] == "Magdalena") & (df["Zafra"] == row["Zafra"]) & (df["Mes"] == row["Mes"])
    if mask.sum() == 1:
        df.loc[mask, col_pol] = row["_pol"]
        df.loc[mask, col_pur] = row["_pur"]
        merged += 1
    elif mask.sum() == 0:
        print(f"  WARNING: No match in Base Maestra for Magdalena {row['Zafra']} {row['Mes']}")
    else:
        print(f"  WARNING: Multiple matches for Magdalena {row['Zafra']} {row['Mes']}")

print(f"Magdalena: {merged} records filled from Mensual_datos")

# STEP 4: Build catorcena-to-month mapping for seasons 2023-2024 and 2024-2025.
# For each month, select the catorcena whose closing date is closest to the 15th.
# Catorcenas without a closing date are skipped (season not started).
cat_fecha = pd.read_excel("../../Magdalena_datos Cengicaña.xlsx", sheet_name="Catorcena_fecha", header=1)
cat_fecha.columns = ['_', 'Catorcena', 'Fecha_Cierre', 'Zafra', 'Pol', 'Pureza']
cat_fecha = cat_fecha[['Catorcena', 'Fecha_Cierre', 'Zafra']].copy()

cat_fecha['Fecha_Cierre'] = pd.to_datetime(cat_fecha['Fecha_Cierre'], dayfirst=True, errors='coerce')
cat_fecha = cat_fecha.dropna(subset=['Fecha_Cierre'])
cat_fecha['Mes_num'] = cat_fecha['Fecha_Cierre'].dt.month
cat_fecha['Dist_15'] = abs(cat_fecha['Fecha_Cierre'].dt.day - 15)

idx = cat_fecha.groupby(['Zafra', 'Mes_num'])['Dist_15'].idxmin()
mapeo_catorcena = cat_fecha.loc[idx].copy()

meses = {1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio',
         7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}
mapeo_catorcena['Mes'] = mapeo_catorcena['Mes_num'].map(meses)

print(f"Catorcena-to-month mapping: {len(mapeo_catorcena)} records")

# STEP 5: Fill Magdalena (seasons 2023-2024 and 2024-2025) from Catorcena_datos.
# Pol values are divided by 10 to match the scale of other seasons.
# If the catorcena has no data (e.g. May), the value stays as NaN.
cat_datos = pd.read_excel("../../Magdalena_datos Cengicaña.xlsx", sheet_name="Catorcena_datos", header=1)
cat_datos.columns = cat_datos.columns.str.strip()
cat_datos['Zafra'] = cat_datos['Zafra'].str.strip()

# Normalize Pol scale
cat_datos['Pol (kg/t)'] = cat_datos['Pol (kg/t)'] / 10

# Standardize season format: "23-24" -> "2023-2024"
cat_datos['Zafra'] = cat_datos['Zafra'].apply(lambda z: f"20{z[:2]}-20{z[3:]}")
mapeo_catorcena['Catorcena_key'] = 'C' + mapeo_catorcena['Catorcena'].astype(int).astype(str)

# Join mapping with catorcena data by season and catorcena number
mapeo_con_datos = mapeo_catorcena.merge(
    cat_datos[['Zafra', 'Catorcena', 'Pol (kg/t)', 'Pureza jugo']],
    left_on=['Zafra', 'Catorcena_key'],
    right_on=['Zafra', 'Catorcena'],
    how='left'
)

merged2 = 0
for _, row in mapeo_con_datos.iterrows():
    if pd.isna(row['Pol (kg/t)']):
        print(f"  WARNING: No data in Catorcena_datos for Magdalena {row['Zafra']} C{int(row['Catorcena_x'])} — left as NaN")
        continue
    mask = (df['Ingenio'] == 'Magdalena') & (df['Zafra'] == row['Zafra']) & (df['Mes'] == row['Mes'])
    if mask.sum() == 1:
        df.loc[mask, col_pol] = row['Pol (kg/t)']
        df.loc[mask, col_pur] = row['Pureza jugo']
        merged2 += 1
    elif mask.sum() == 0:
        print(f"  WARNING: No match in Base Maestra for Magdalena {row['Zafra']} {row['Mes']}")

print(f"Magdalena: {merged2} records filled from Catorcena_datos")

# ── END OF CLEANING ───────────────────────────────────────────────────────────

# Save Base maestra
df.to_excel(OUTPUT_PATH, index=False)
print(f"File saved: {OUTPUT_PATH}")
print(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")

# Export compiled Magdalena data as standalone file (independent from Base maestra)
# Part 1: monthly data (seasons 2020-2021 to 2022-2023)
mag_mensual = ceng[ceng['Ingenio'] == 'Magdalena'][['Ingenio', 'Zafra', 'Mes', '_pol', '_pur']].copy()
mag_mensual = mag_mensual.rename(columns={'_pol': 'Pol (kg/t)', '_pur': 'Pureza jugo'})

# Part 2: catorcena data mapped to months (seasons 2023-2024 and 2024-2025)
mag_catorcena = mapeo_con_datos[['Zafra', 'Mes', 'Pol (kg/t)', 'Pureza jugo']].copy()
mag_catorcena.insert(0, 'Ingenio', 'Magdalena')

magdalena_compilado = pd.concat([mag_mensual, mag_catorcena], ignore_index=True)
magdalena_compilado.to_excel("../Magdalena_datos_filtrados.xlsx", index=False)
print(f"Magdalena exported: {magdalena_compilado.shape[0]} rows -> Magdalena_datos_filtrados.xlsx")
