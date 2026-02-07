#!/usr/bin/env python3
"""Export dashboard-ready artifacts from nyc_housing_cleaned.csv

Produces:
 - nyc_housing_cleaned.parquet
 - samples: sample_overall.csv, sample_by_borough.csv
 - aggregates: median_price_by_zip.csv, median_price_by_borough.csv
 - schema: schema.json
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np


ROOT = Path(__file__).parent
IN_CSV = ROOT / 'nyc_housing_cleaned.csv'


def build_schema(df: pd.DataFrame):
    types = {}
    for c in df.columns:
        dt = str(df[c].dtype)
        types[c] = dt
    return types


def main():
    if not IN_CSV.exists():
        print('Cleaned CSV not found:', IN_CSV)
        return

    df = pd.read_csv(IN_CSV)

    out_parquet = ROOT / 'nyc_housing_cleaned.parquet'
    try:
        df.to_parquet(out_parquet, index=False)
        print('Wrote', out_parquet)
    except Exception as e:
        print('Could not write parquet (pyarrow/fastparquet missing). Skipping parquet. Error:', e)

    # Overall sample (random)
    sample_overall = ROOT / 'sample_overall.csv'
    df.sample(n=min(5000, len(df)), random_state=1).to_csv(sample_overall, index=False)
    print('Wrote', sample_overall)

    # Sample by borough (up to 1000 each)
    sample_borough = ROOT / 'sample_by_borough.csv'
    parts = []
    for b, g in df.groupby('borough'):
        parts.append(g.sample(n=min(1000, len(g)), random_state=1))
    pd.concat(parts, ignore_index=True).to_csv(sample_borough, index=False)
    print('Wrote', sample_borough)

    # Aggregates
    if 'zip_code' in df.columns:
        agg_zip = df.groupby('zip_code', dropna=True)['sale_price'].median().reset_index()
        agg_zip = agg_zip.rename(columns={'sale_price': 'median_sale_price'})
        agg_zip.to_csv(ROOT / 'median_price_by_zip.csv', index=False)
        print('Wrote median_price_by_zip.csv')

    agg_borough = df.groupby('borough')['sale_price'].median().reset_index().rename(columns={'sale_price': 'median_sale_price'})
    agg_borough.to_csv(ROOT / 'median_price_by_borough.csv', index=False)
    print('Wrote median_price_by_borough.csv')

    # Price per sqft (sale_price / bldgarea) where bldgarea>0
    df['price_per_bldg_sqft'] = np.where(df['bldgarea'] > 0, df['sale_price'] / df['bldgarea'], np.nan)
    df[['borough','block','lot','sale_price','bldgarea','price_per_bldg_sqft']].head().to_csv(ROOT / 'price_per_sqft_sample.csv', index=False)
    print('Wrote price_per_sqft_sample.csv')

    # Schema
    schema = build_schema(df)
    with open(ROOT / 'schema.json', 'w') as f:
        json.dump(schema, f, indent=2)
    print('Wrote schema.json')


if __name__ == '__main__':
    main()
