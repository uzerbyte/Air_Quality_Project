"""Check saved results, test predictions and invalid inputs."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import joblib,nbformat
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from features import prepare_features,predict_records,RAW_COLUMNS,SENSORS
root=Path(__file__).resolve().parent
n=nbformat.read(root/'Air_Quality_Analysis.ipynb',as_version=4)
code=[c for c in n.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code)
assert not any(o.output_type=='error' for c in code for o in c.outputs)
assert sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)>=7
raw=root/'data/raw/AirQualityUCI.csv'
assert hashlib.sha256(raw.read_bytes()).hexdigest()=='13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996'
bundle=joblib.load(root/'artifacts/air_quality_model.joblib')
t=pd.read_csv(root/'artifacts/test_predictions.csv')
yhat=predict_records(bundle,t)['predicted_CO_mg_m3'].to_numpy()
np.testing.assert_allclose(yhat,t.prediction,rtol=1e-10,atol=1e-10)
m=pd.read_csv(root/'artifacts/test_metrics.csv').iloc[1]
np.testing.assert_allclose([mean_absolute_error(t['CO(GT)'],yhat),mean_squared_error(t['CO(GT)'],yhat)**.5,r2_score(t['CO(GT)'],yhat)],[m.MAE,m.RMSE,m.R2])
assert pd.to_datetime(t.timestamp).min()>pd.Timestamp(bundle['training_end'])
x=t.head(1).copy(); x['CO(GT)']=1e10
assert not any('(GT)' in c for c in prepare_features(x).columns)
x.loc[:,SENSORS]=-200
assert predict_records(bundle,x).status.iloc[0].endswith('abstain')
for broken in [x.drop(columns='T'),x.assign(timestamp='invalid-date')]:
 try:predict_records(bundle,broken)
 except (ValueError,KeyError):pass
 else:raise AssertionError('Invalid input should fail explicitly')
print(f'PASS: {len(code)} executed code cells, source integrity, saved-model parity, test metrics, chronology, leakage exclusion and invalid/outage input checks.')
