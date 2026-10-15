"""Create sensor, weather and time features for CO prediction."""
import numpy as np
import pandas as pd

SENSORS=['PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)']
WEATHER=['T','RH']
RAW_COLUMNS=['timestamp']+SENSORS+WEATHER

def prepare_features(frame, recipe='engineered'):
    """Create same-hour inputs without reference pollutant measurements."""
    result=frame.copy()
    time=pd.to_datetime(result['timestamp'],errors='raise')
    for column in SENSORS+WEATHER:
        result[column]=pd.to_numeric(result[column],errors='coerce').replace(-200,np.nan)
    result.loc[(result.RH<0)|(result.RH>100),'RH']=np.nan
    for column in SENSORS:
        result.loc[result[column]<0,column]=np.nan
    result['hour']=time.dt.hour.astype(str)
    result['weekday']=time.dt.dayofweek.astype(str)
    result['month']=time.dt.month.astype(str)
    if recipe=='raw':
        result['hour_numeric']=time.dt.hour
        return result[SENSORS+WEATHER+['hour_numeric']]
    result['hour_sin']=np.sin(2*np.pi*time.dt.hour/24)
    result['hour_cos']=np.cos(2*np.pi*time.dt.hour/24)
    result['year_sin']=np.sin(2*np.pi*(time.dt.dayofyear-1)/365.25)
    result['year_cos']=np.cos(2*np.pi*(time.dt.dayofyear-1)/365.25)
    result['weekend']=(time.dt.dayofweek>=5).astype(int)
    result['temperature_squared']=result['T']**2
    result['S1_x_humidity']=result[SENSORS[0]]*result.RH/100
    common=WEATHER+['hour_sin','hour_cos','year_sin','year_cos','weekend','temperature_squared']
    if recipe=='weather':return result[common+['weekday']]
    return result[SENSORS+common+['S1_x_humidity','weekday']]

def predict_records(bundle, records):
    """Predict CO and flag rows with too few sensor readings."""
    frame=pd.DataFrame(records).copy()
    missing=set(RAW_COLUMNS)-set(frame.columns)
    if missing:raise ValueError('Missing required columns: '+', '.join(sorted(missing)))
    inputs=prepare_features(frame,bundle['recipe'])
    output=pd.DataFrame({'timestamp':frame['timestamp'].astype(str)})
    output['predicted_CO_mg_m3']=bundle['model'].predict(inputs)
    available=frame[SENSORS].apply(pd.to_numeric,errors='coerce').replace(-200,np.nan).notna().sum(axis=1)
    output['status']=np.where(available>=3,'research estimate','insufficient sensor coverage - abstain')
    return output
