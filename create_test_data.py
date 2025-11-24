import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------
# 未来予測用データの日数
FUTURE_DAYS = 30

# 学習用ダミーデータの日数（過去データ用）
PAST_DAYS = 365

# ---------------------------------------------------------
# 1. 未来予測用データの作成 (sample_future_features.csv)
# ---------------------------------------------------------
# アプリの「未来予測モード」でアップロードするためのファイルです。
# ランダムフォレストが予測するために必要な「未来の特徴量(Lagなど)」を含みます。

print("未来予測用データを作成中...")

# 明日から30日分の日付
start_date = datetime.now() + timedelta(days=1)
date_range = [start_date + timedelta(days=x) for x in range(FUTURE_DAYS)]

df_future = pd.DataFrame({
    'Date': date_range
})

# 特徴量エンジニアリング（ダミー値）
# ※実際は直近の実績から計算する必要がありますが、テスト用なので乱数で埋めます
df_future['Lag_1'] = np.random.randint(1000, 5000, size=FUTURE_DAYS) # 1日前の売上想定
df_future['Lag_7'] = np.random.randint(1000, 5000, size=FUTURE_DAYS) # 7日前の売上想定
df_future['DayOfWeek'] = df_future['Date'].dt.dayofweek # 曜日（0:月曜 ~ 6:日曜）

# 日付を文字列に変換（保存用）
df_future['Date'] = df_future['Date'].dt.strftime('%Y-%m-%d')

# CSV保存
df_future.to_csv('sample_future_features.csv', index=False)
print("'sample_future_features.csv' を保存しました。")