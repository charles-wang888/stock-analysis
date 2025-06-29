import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib
matplotlib.use('Agg')  # 或 'TkAgg'

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

def load_data():
    # 加载数据
    df_price = pd.read_csv("01data.csv", parse_dates=["date"])
    df_features = pd.read_csv("selected_features_with_data.csv", parse_dates=["date"])

    # 合并数据
    df_merged = pd.merge(df_price[['date', 'close']], df_features, on='date')

    # 创建目标变量（未来5天的收盘价）
    df_merged['target'] = df_merged['close'].shift(-5)

    # 删除最后5行（因为无法计算目标变量）
    df_merged = df_merged.dropna()

    return df_merged

def prepare_features(df):
    # 分离特征和目标变量
    feature_cols = [col for col in df.columns if col not in ['date', 'close', 'target']]
    X = df[feature_cols]
    y = df['target']

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 保存scaler供后续使用
    joblib.dump(scaler, 'scaler.pkl')

    return X_scaled, y, feature_cols

def train_model(X, y):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # 训练随机森林回归模型
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n模型评估报告：")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"均方根误差 (RMSE): {np.sqrt(mse):.2f}")
    print(f"决定系数 (R²): {r2:.4f}")

    # 保存模型
    joblib.dump(model, 'stock_predictor.pkl')

    return model, X_test, y_test, y_pred

def plot_results(df, y_test, y_pred):
    # 获取测试集的日期
    test_dates = df['date'].iloc[-len(y_test):]

    # 创建预测结果图
    plt.figure(figsize=(15, 7))
    plt.plot(test_dates, y_test, label='实际价格', color='blue')
    plt.plot(test_dates, y_pred, label='预测价格', color='red', linestyle='--')
    plt.title('股票价格预测结果', fontsize=12)
    plt.xlabel('日期', fontsize=10)
    plt.ylabel('价格', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载数据
    print("正在加载数据...")
    df = load_data()

    # 准备特征
    print("正在准备特征...")
    X, y, feature_cols = prepare_features(df)

    # 训练模型
    print("正在训练模型...")
    model, X_test, y_test, y_pred = train_model(X, y)

    # 绘制结果
    print("正在生成预测结果图...")
    plot_results(df, y_test, y_pred)

    print("\n预测模型已训练完成并保存。")
    print("模型文件：stock_predictor.pkl")
    print("特征标准化器：scaler.pkl")
    print("预测结果图：prediction_results.png")

if __name__ == "__main__":
    main()