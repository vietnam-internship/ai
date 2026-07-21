from ai.data_preprocessing.preprocess import fetch_and_fill_exchange_rate_timeseries
from ai.baseline.baseline import build_baseline_dataset


def load_actual_and_predicted():
    df = fetch_and_fill_exchange_rate_timeseries()
    baseX, basey = build_baseline_dataset(df)
    actual = basey
    predicted = baseX['rate_pred']
    return actual, predicted


#(|실제 관측값 - 모델의 예측값| / 실제 관측값 의 합) / 데이터 개수 * 100
def mape():
    actual, predicted = load_actual_and_predicted()
    ape = (actual - predicted).abs() / actual
    return ape.mean() * 100


#|실제 관측값 - 모델의 예측값| 합
def mae():
    actual, predicted = load_actual_and_predicted()
    return (actual - predicted).abs().sum()
