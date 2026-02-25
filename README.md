# Molasses Exhaustion Analysis — Guatemalan Sugar Mills

Multi-harvest analysis of final molasses composition and production efficiency from 8 Guatemalan sugar mills (2020–2025).

## Overview

This project investigates sugar exhaustion patterns by analyzing the relationship between production metrics, molasses composition, and mill efficiency across 5 harvest seasons. Focus areas include:

- **Temporal evolution** of key quality ratios (AR/C, Purity) per mill
- **Seasonal patterns** in composition throughout the harvest
- **Loss accounting** — analytical methods vs. actual recovery
- **Viscosity and carbohydrate destruction** as exhaustion predictors

## Dataset

A single clean dataset (`data/processed/Base maestra.xlsx`) serves as the starting point for all analysis. It contains **248 samples** from 8 mills across 5 harvest seasons (November–May), linking physicochemical composition with monthly production metrics.

| Variable group | Variables |
|---|---|
| Identity | Zafra, Mes, Ingenio |
| Chemical composition | Brix, Pol%, Sacarosa HPLC, Glucosa, Fructosa, AR Fehling, AR HPLC |
| Quality ratios | Pureza Pol, Pureza Clerget, Pureza Real, AR/Cenizas |
| Physical properties | Viscosidad 25°C, Viscosidad 40°C, Color, pH |
| Production metrics | Pol en caña, Pureza jugo, Recuperación Total, kg/t MF |
| Derived | Fructosa/Glucosa, Perd_Indet, Mej_alcanzable_PUR |

> Raw data is not included in this repository.

## Project Structure

```
molasses-exhaustion-analysis/
├── data/
│   └── processed/          # clean dataset (gitignored)
├── scripts/                # analysis scripts
│   ├── 01_temporal.py
│   ├── 02_seasonal.py
│   └── 03_losses.py
└── outputs/
    └── figures/            # saved plots
```

## Analysis Framework

### 1. Temporal Evolution (Multi-Harvest)
- AR/C ratio consistency per mill across 5 zafras
- Pureza Objetivo Difference (DPO) trends
- Year-over-year mill improvement or deterioration

### 2. Seasonal Patterns
- Purity and AR/ceniza changes across harvest phases (early, mid, late)
- Correlation with Pol%caña and final molasses yield (kg/t)
- Within-harvest vs. cross-mill variation

### 3. Loss Analysis
- Systematic error: Pol vs. Sacarosa HPLC
- Indeterminate loss estimation by analytical method
- Mill-to-mill loss profiles

### 4. Composition-Level Drivers
- **Viscosity (40°C)**: Predictive threshold for exhaustion difficulty
- **F/G ratio**: Glucose destruction indicator and Maillard correlation with Color
- **AR/C validation**: Relationship to actual juice purity

## Authors

Rodrigo Padilla — rodrigopadilla267@gmail.com
Raisa Vega
