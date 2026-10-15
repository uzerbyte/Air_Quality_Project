# Analysing Air Quality and Its Link to Urban Factors
DSAI501, Option B. Same-hour carbon monoxide estimation using sensor, weather and time data.

## Run
Use Python 3.12.14. From this folder:
```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name air-quality --display-name "Air Quality"
```
Open `Air_Quality_Analysis.ipynb` in Jupyter or VS Code, select this kernel, and Run All from the project directory. The delivered notebook is already executed with inline figures and numerical outputs. To execute without a notebook UI:
```sh
python -c "import nbformat; from nbclient import NotebookClient; p='Air_Quality_Analysis.ipynb'; n=nbformat.read(p,as_version=4); NotebookClient(n,kernel_name='air-quality',timeout=600).execute(); nbformat.write(n,p)"
python verify_project.py
python predict.py --input artifacts/demo_input.csv
```
Full execution overwrites generated `artifacts/` and `figures/`. Model search can take several minutes. The persisted model uses scikit-learn 1.9.1; load only trusted joblib files with matching package versions.

## Contents
- Executed notebook: full audit, preprocessing, EDA, chronological validation, six candidate models, final test, diagnostics and export.
- `features.py`, `predict.py`: input preparation and batch prediction.
- `artifacts/air_quality_model.joblib`: fitted log-target ridge pipeline, trained only on pre-test training and validation records. `model_metadata.json` records input schema and versions.
- `data/raw/`: original UCI CSV. `data/processed/`: cleaned audit-ready observations.
- `artifacts/`: saved model, predictions, metrics and diagnostics.
- `figures/`: chart images created when the notebook is run; charts are also embedded in the notebook.

## Interpretation
This is same-hour CO reference concentration estimation (mg/m³), not forecasting. Calendar patterns are urban activity proxies; traffic volumes, land use, wind and rainfall are unavailable. Test RMSE is 0.676 mg/m³ and R² 0.757. These are research results from one historical site. Predictions marked “insufficient sensor coverage - abstain” must not support decisions. Even adequate-coverage estimates require local validation and reference measurement confirmation.

## Dataset
Vito, S. (2008). *Air quality*. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F

Accessed 15 October 2026. The CSV uses semicolons and decimal commas. Remove empty export rows and columns, parse timestamps, and convert -200 to missing values. Exclude missing CO targets from modelling. Fit predictor imputation and scaling within the training pipelines. Original CSV SHA-256: `13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996`.
