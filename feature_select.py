from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
matplotlib.use('Agg')  # 或 'TkAgg'

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

def main():
    # 加载数据
    df = pd.read_csv("01data.csv")
    df['date'] = pd.to_datetime(df['date'])
    df_all_features = pd.read_csv("02data.csv")
    df_all_features['date'] = pd.to_datetime(df_all_features['date'])

    # 加载特征名称
    feature_names = pd.read_csv("feature_names.csv")
    feature_name_dict = dict(zip(feature_names['feature_name'], feature_names['description']))

    X = df_all_features.drop(columns=["date"])
    df_label = df[['date', 'close']].copy()
    df_label['target'] = (df_label['close'].shift(-5) > df_label['close']).astype(int)
    y = df_label[df_label['date'].isin(df_all_features['date'])]['target'].reset_index(drop=True)

    impute(X)
    X_selected = select_features(X, y)

    # 使用随机森林计算特征重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_selected, y)

    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'feature_name': X_selected.columns,
        'importance': rf.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    # 添加特征描述
    feature_importance['description'] = feature_importance['feature_name'].map(feature_name_dict)

    # 保存特征重要性
    feature_importance.to_csv('feature_importance.csv', index=False)

    # 绘制特征重要性图
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_importance)), feature_importance['importance'])
    plt.xticks(range(len(feature_importance)),
               feature_importance['description'],
               rotation=45,
               ha='right',
               fontsize=8)
    plt.title('特征重要性排序', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n选出的有效因子数：{X_selected.shape[1]}")
    print("\n前10个重要特征：")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"{row['description']}: {row['importance']:.4f}")

    # 保存选中的特征数据
    df_selected = df_all_features[['date'] + X_selected.columns.tolist()]
    df_selected.to_csv("selected_features_with_data.csv", index=False)

if __name__ == '__main__':
    main()