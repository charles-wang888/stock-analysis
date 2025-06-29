import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
df_price = pd.read_csv("01data.csv", parse_dates=["date"])
df_features = pd.read_csv("selected_features_with_data.csv", parse_dates=["date"])
df_merged = pd.merge(df_price[['date', 'close']], df_features, on='date')

# 加载特征重要性数据
feature_importance = pd.read_csv("feature_importance.csv")
# 按重要性排序特征
feature_importance = feature_importance.sort_values('importance', ascending=False)
# 获取前10个重要特征
top_features = feature_importance.head(10)['feature_name'].tolist()

# 创建图表
fig, ax1 = plt.subplots(figsize=(16, 8))
ax1.set_xlabel("日期")
ax1.set_ylabel("收盘价", color='tab:blue')
ax1.plot(df_merged['date'], df_merged['close'], color='tab:blue', linewidth=2, label='收盘价')
ax2 = ax1.twinx()

# 为每个特征选择不同的颜色
colors = plt.cm.tab10.colors

# 绘制前10个重要特征
for i, feature in enumerate(top_features):
    # 获取特征描述
    feature_desc = feature_importance[feature_importance['feature_name'] == feature]['description'].iloc[0]
    # 绘制特征线
    ax2.plot(df_merged['date'], df_merged[feature],
             linestyle='--',
             label=f'{feature_desc} ({feature})',
             color=colors[i % len(colors)])

# 设置图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='upper left',
           bbox_to_anchor=(1.15, 1),
           fontsize=8)

plt.title("收盘价 + 重要因子走势", fontsize=16)
plt.grid(True)
plt.tight_layout()

# 保存图表
plt.savefig('price_and_features.png', bbox_inches='tight', dpi=300)
plt.close()

print("图表已保存为 price_and_features.png")