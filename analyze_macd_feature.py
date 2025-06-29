import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
from tsfresh.feature_extraction import feature_calculators
import matplotlib
matplotlib.use('Agg')  # 或 'TkAgg'

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

def calculate_macd_features(df):
    # 计算MACD
    macd, macdsignal, macdhist = talib.MACD(df['close'], 12, 26, 9)

    # 计算Friedrich系数
    # 添加所有必需的参数：coeff, m, r
    friedrich_coef = feature_calculators.friedrich_coefficients(
        macd,
        param=[{
            "coeff": 3,  # 系数阶数
            "m": 3,      # 移动窗口大小
            "r": 30      # 时间窗口大小
        }]
    )

    return {
        'macd': macd,
        'macd_signal': macdsignal,
        'macd_hist': macdhist,
        'friedrich_coef': friedrich_coef
    }

def plot_macd_analysis(df, features):
    plt.figure(figsize=(15, 10))

    # 绘制价格和MACD
    ax1 = plt.subplot(211)
    ax1.plot(df['date'], df['close'], label='收盘价')
    ax1.set_title('价格走势', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    ax2 = plt.subplot(212)
    ax2.plot(df['date'], features['macd'], label='MACD')
    ax2.plot(df['date'], features['macd_signal'], label='MACD Signal')
    ax2.bar(df['date'], features['macd_hist'], label='MACD Histogram')
    ax2.set_title('MACD指标', fontsize=12)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('macd_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 打印Friedrich系数
    print("\nMACD Friedrich系数分析：")
    print(f"Friedrich系数值: {features['friedrich_coef']}")
    print("\nFriedrich系数的含义：")
    print("1. 系数值越大，表示MACD的趋势越强")
    print("2. 系数值接近0，表示MACD处于震荡状态")
    print("3. 系数值的变化可以反映MACD趋势的转折点")
    print("\n参数说明：")
    print("- coeff: 3 (系数阶数)")
    print("- m: 3 (移动窗口大小)")
    print("- r: 30 (时间窗口大小)")

def main():
    # 加载数据
    df = pd.read_csv("01data.csv")
    df['date'] = pd.to_datetime(df['date'])

    # 计算特征
    features = calculate_macd_features(df)

    # 绘制分析图
    plot_macd_analysis(df, features)

if __name__ == "__main__":
    main()