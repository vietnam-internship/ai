#오늘까지의 평균값을 내일 환율이라고 가정하고, 그게 얼마나 틀렸는지 확인하는 baseline
from ai.data_preprocessing.feature import calculate_moving_average, calculate_moving_std
from ai.data_preprocessing.model_input import build_model_dataset

def check_row_num(df):
    if len(df)<30:
        return None
    else:
        return df
#df의 column : 오늘 환율, 이동평균 환율(=30일 간의 환율 평균), 이동표준편차, 그 다음날 환율
def expansion_df(df):
    if check_row_num(df) is not None:
        df = calculate_moving_average(df, window=30)
        df = calculate_moving_std(df, window=30)
        df['rate_pred']=df['rate_ma30']
        df['actual_rate'] = df['rate'].shift(-1)
        df = df.dropna()
        return df
    else:
        raise ValueError("데이터가 충분하지 않습니다.")
    
def build_baseline_dataset(df):
    df = expansion_df(df)
    X,y = build_model_dataset(df, 'actual_rate',['rate_pred'],horizon=0)
    return X,y
