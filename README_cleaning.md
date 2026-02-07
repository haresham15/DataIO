Cleaning instructions for nyc_housing dataset

1. Create a Python virtual environment and install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the cleaner (default reads `nyc_housing_base.csv` in the same folder):

```bash
python clean_nyc_housing.py
```

3. Output: `nyc_housing_cleaned.csv` in the same folder and a short report printed to stdout.

If you want to pass a different input file: `python clean_nyc_housing.py path/to/file.csv`
