from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
from tqdm import tqdm
import pandas as pd

df = pd.read_csv("01data.csv")
df['date'] = pd.to_datetime(df['date'])

WINDOW_SIZE = 20
features_list = []

for i in tqdm(range(WINDOW_SIZE, len(df))):
    df_window = df.iloc[i - WINDOW_SIZE:i].copy()
    tsfresh_df = df_window[['date', 'close', 'volume', 'rsi', 'macd']].copy()
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

    # 保留原始特征名称，不再重命名为feature_0, feature_1等
    features['date'] = df.iloc[i]['date']
    features_list.append(features)

df_all_features = pd.concat(features_list).reset_index(drop=True)

# 保存特征名称到文件
feature_names = pd.DataFrame({
    'feature_name': df_all_features.columns.tolist(),
    'description': ['特征' + str(i) for i in range(len(df_all_features.columns))]
})
feature_names.to_csv('feature_names.csv', index=False)

# 保存特征数据
df_all_features.to_csv("02data.csv", index=False)

print(f"总共提取了 {len(df_all_features.columns)} 个特征")
print("特征名称已保存到 feature_names.csv")