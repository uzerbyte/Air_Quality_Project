"""Predict CO from a CSV using the saved model."""
import argparse
import joblib
import pandas as pd
from features import predict_records
parser=argparse.ArgumentParser()
parser.add_argument('--input',default='artifacts/demo_input.csv')
parser.add_argument('--model',default='artifacts/air_quality_model.joblib')
parser.add_argument('--output',default='artifacts/new_predictions.csv')
args=parser.parse_args()
result=predict_records(joblib.load(args.model),pd.read_csv(args.input))
result.to_csv(args.output,index=False)
print(result.to_string(index=False))
print('Saved:',args.output)
