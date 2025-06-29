import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import akshare as ak
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
import matplotlib
matplotlib.use('Agg')  # 或 'TkAgg'

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

def get_stock_data(symbol, start_date, end_date):
    """获取股票数据"""
    df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start_date, end_date=end_date)
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume"
    })
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)

def extract_tsfresh_features(df, window_size=20):
    """使用tsfresh提取特征"""
    features_list = []

    for i in range(window_size, len(df)):
        df_window = df.iloc[i - window_size:i].copy()
        tsfresh_df = df_window[['date', 'close', 'volume']].copy()
        tsfresh_df['id'] = 0
        df_long = tsfresh_df.melt(id_vars=["date", "id"], var_name="kind", value_name="value")
        df_long.rename(columns={"date": "time"}, inplace=True)

        features = extract_features(
            df_long,
            column_id="id", column_sort="time", column_kind="kind",
            column_value="value", default_fc_parameters=EfficientFCParameters(),
            n_jobs=0, disable_progressbar=True
        )
        impute(features)
        features['date'] = df.iloc[i]['date']
        features_list.append(features)

    return pd.concat(features_list).reset_index(drop=True)

def prepare_features(df):
    """准备特征数据"""
    # 提取tsfresh特征
    print("正在提取特征...")
    df_features = extract_tsfresh_features(df)

    # 加载特征选择结果
    selected_features = pd.read_csv("selected_features_with_data.csv")
    feature_cols = [col for col in selected_features.columns if col != 'date']

    # 确保所有需要的特征都存在
    missing_features = set(feature_cols) - set(df_features.columns)
    if missing_features:
        print(f"警告：缺少以下特征：{missing_features}")
        # 为缺失的特征添加0值
        for feature in missing_features:
            df_features[feature] = 0

    # 只保留需要的特征
    X = df_features[feature_cols]

    # 加载标准化器并转换数据
    scaler = joblib.load('scaler.pkl')
    X_scaled = scaler.transform(X)

    return X_scaled, df_features['date']

def evaluate_predictions(actual, predicted):
    """评估预测结果"""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    print("\n预测评估指标：")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"决定系数 (R²): {r2:.4f}")

    # 计算预测准确率（预测方向正确的比例）
    direction_accuracy = np.mean((np.diff(actual) > 0) == (np.diff(predicted) > 0))
    print(f"方向准确率: {direction_accuracy:.2%}")

    # 计算平均相对误差
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print(f"平均相对误差 (MAPE): {mape:.2f}%")

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_accuracy': direction_accuracy,
        'mape': mape
    }

def plot_predictions(dates, actual, predicted, symbol):
    """绘制预测结果对比图"""
    plt.figure(figsize=(15, 8))
    plt.plot(dates, actual, label='实际价格', color='blue')
    plt.plot(dates, predicted, label='预测价格', color='red', linestyle='--')
    plt.title(f'{symbol} 股票价格预测对比', fontsize=12)
    plt.xlabel('日期', fontsize=10)
    plt.ylabel('价格', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('prediction_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    symbol = "300348"  # 长亮科技

    # 设置时间范围（最近3个月）
    end_date = datetime.today().strftime('%Y%m%d')
    start_date = (datetime.today() - timedelta(days=90)).strftime('%Y%m%d')

    print(f"正在获取 {symbol} 的历史数据...")
    df = get_stock_data(symbol, start_date, end_date)

    print("正在准备特征...")
    X, dates = prepare_features(df)

    print("正在加载模型...")
    model = joblib.load('stock_predictor.pkl')

    print("正在进行预测...")
    predictions = model.predict(X)

    # 获取实际价格（从原始数据中获取）
    actual_prices = df['close'].values[20:]  # 跳过前20天，因为需要这些数据来提取特征

    # 评估预测结果
    metrics = evaluate_predictions(actual_prices, predictions)

    # 绘制预测结果
    plot_predictions(dates, actual_prices, predictions, symbol)

    # 保存预测结果
    results_df = pd.DataFrame({
        'date': dates,
        'actual_price': actual_prices,
        'predicted_price': predictions
    })
    results_df.to_csv('prediction_results.csv', index=False)

    print("\n预测结果已保存到 prediction_results.csv")
    print("预测评估图表已保存到 prediction_evaluation.png")

if __name__ == "__main__":
    main()