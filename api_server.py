from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
import pandas as pd


ROOT = Path(__file__).parent
DATA_PARQUET = ROOT / 'nyc_housing_cleaned.parquet'
DATA_CSV = ROOT / 'nyc_housing_cleaned.csv'


app = FastAPI(title='NYC Housing API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def load_data() -> pd.DataFrame:
    """Load the cleaned dataset (parquet preferred)."""
    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
    elif DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
    else:
        raise FileNotFoundError('No cleaned dataset found')
    return df


@app.on_event('startup')
def startup():
    try:
        app.state.df = load_data()
        app.state.total = len(app.state.df)
    except Exception as e:
        # keep app up but endpoints will return errors
        app.state.df = None
        app.state.total = 0
        app.state.load_error = str(e)


@app.get('/api/rows')
def get_rows(limit: int = 100, offset: int = 0):
    """Return paginated rows."""
    if app.state.df is None:
        raise HTTPException(status_code=500, detail=f'data not loaded: {getattr(app.state, "load_error", "unknown")}')
    df = app.state.df
    slice_df = df.iloc[offset: offset + limit]
    return {
        'total': app.state.total,
        'offset': offset,
        'limit': limit,
        'rows': jsonable_encoder(slice_df.where(pd.notnull(slice_df), None).to_dict(orient='records'))
    }


@app.get('/api/sample')
def get_sample(borough: Optional[str] = Query(None), limit: int = 500):
    """Return a sample (optionally by borough)."""
    if app.state.df is None:
        raise HTTPException(status_code=500, detail='data not loaded')
    df = app.state.df
    if borough:
        df = df[df['borough'] == borough]
    if len(df) == 0:
        return {'rows': []}
    sample = df.sample(n=min(limit, len(df)), random_state=1)
    return {'rows': jsonable_encoder(sample.where(pd.notnull(sample), None).to_dict(orient='records'))}


@app.get('/api/aggregate')
def get_aggregate(groupby: str = Query('zip_code'), metric: str = Query('median')):
    """Aggregate sale_price by `groupby` (zip_code, borough).

    metric: median | mean | count
    """
    if app.state.df is None:
        raise HTTPException(status_code=500, detail='data not loaded')
    df = app.state.df
    if groupby not in df.columns:
        raise HTTPException(status_code=400, detail=f'groupby must be a column in data')
    if metric == 'median':
        res = df.groupby(groupby)['sale_price'].median().reset_index().rename(columns={'sale_price': 'median_sale_price'})
    elif metric == 'mean':
        res = df.groupby(groupby)['sale_price'].mean().reset_index().rename(columns={'sale_price': 'mean_sale_price'})
    elif metric == 'count':
        res = df.groupby(groupby)['sale_price'].count().reset_index().rename(columns={'sale_price': 'count'})
    else:
        raise HTTPException(status_code=400, detail='unsupported metric')
    return jsonable_encoder(res.where(pd.notnull(res), None).to_dict(orient='records'))


@app.get('/api/geo')
def get_geo(minlon: float, minlat: float, maxlon: float, maxlat: float, limit: int = 1000):
    """Return points within the bbox (minlon,minlat,maxlon,maxlat)."""
    if app.state.df is None:
        raise HTTPException(status_code=500, detail='data not loaded')
    df = app.state.df
    mask = (
        (df['longitude'] >= minlon) & (df['longitude'] <= maxlon) &
        (df['latitude'] >= minlat) & (df['latitude'] <= maxlat)
    )
    sel = df[mask].head(limit)
    return {'count': int(sel.shape[0]), 'rows': jsonable_encoder(sel.where(pd.notnull(sel), None).to_dict(orient='records'))}
