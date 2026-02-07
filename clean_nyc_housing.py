#!/usr/bin/env python3
"""Simple cleaning pipeline for nyc_housing_base.csv

Saves cleaned output to `nyc_housing_cleaned.csv` and prints a small report.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def main(src_path: Path):
    df = pd.read_csv(src_path)

    before_rows = len(df)

    # Rename columns for clarity
    df = df.rename(columns={
        'borough_x': 'borough_code',
        'borough_y': 'borough',
        'sale_price': 'sale_price',
    })

    # Trim whitespace in string columns
    obj_cols = df.select_dtypes(['object']).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # Convert numeric-looking columns
    numeric_cols = ['block','lot','sale_price','zip_code','yearbuilt','lotarea','bldgarea',
                    'resarea','comarea','unitsres','unitstotal','numfloors','latitude','longitude','building_age']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing or non-positive sale price
    missing_price = df['sale_price'].isna().sum()
    df = df[df['sale_price'].notna()]
    df = df[df['sale_price'] > 0]

    # Normalize zip_code to integer where possible
    if 'zip_code' in df.columns:
        df['zip_code'] = df['zip_code'].fillna(0).astype(int).replace({0: np.nan})

    # Keep borough (string) from borough column if present
    if 'borough' in df.columns:
        df['borough'] = df['borough'].replace({'nan': np.nan})

    # Remove rows with invalid coordinates (outside rough NYC bounds)
    lat_ok = df['latitude'].between(40.0, 41.2)
    lon_ok = df['longitude'].between(-74.5, -73.0)
    coord_bad = (~lat_ok) | (~lon_ok)
    coord_bad_count = coord_bad.sum()
    df = df[lat_ok & lon_ok]

    # Drop exact duplicates
    before_dedup = len(df)
    df = df.drop_duplicates()
    dedup_removed = before_dedup - len(df)

    # For multiple records per property (same borough/block/lot), keep the one with max sale_price
    if set(['borough','block','lot']).issubset(df.columns):
        idx = df.groupby(['borough','block','lot'])['sale_price'].idxmax()
        df = df.loc[idx].reset_index(drop=True)

    # Add a log price column for modeling convenience
    df['sale_price_log'] = np.log1p(df['sale_price'])

    # Final counts
    after_rows = len(df)

    out_path = src_path.parent / 'nyc_housing_cleaned.csv'
    df.to_csv(out_path, index=False)

    print('Cleaning report:')
    print(f'  source file: {src_path}')
    print(f'  rows before: {before_rows}')
    print(f'  rows after dropping missing/non-positive price: {before_rows - missing_price if pd.notna(missing_price) else before_rows}')
    print(f'  invalid coords removed: {int(coord_bad_count)}')
    print(f'  exact duplicate rows removed: {int(dedup_removed)}')
    print(f'  rows after cleaning: {after_rows}')
    print(f'  cleaned file written to: {out_path}')


if __name__ == '__main__':
    src = Path(__file__).parent / 'nyc_housing_base.csv'
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    if not src.exists():
        print('Source CSV not found:', src)
        sys.exit(2)
    main(src)
