import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

# -------------------------------------------
# 1. ページ設定
# -------------------------------------------
st.set_page_config(
    page_title="需要予測ダッシュボード",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("需要予測ダッシュボード")
st.markdown("売上の可視化と予測を行うダッシュボードアプリケーションです")

# -------------------------------------------
# 2. データ読み込み
# -------------------------------------------
@st.cache_data
def load_data():
    try:
        df_rich = pd.read_csv('daily_sales_rich.csv')
        df_rich['Date'] = pd.to_datetime(df_rich['Date'])
    except FileNotFoundError:
        st.error("Error: daily_sales_rich.csv not found.")
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        df_sku = pd.read_csv('top_sku_daily.csv')
        df_sku['Date'] = pd.to_datetime(df_sku['Date'])
    except FileNotFoundError:
        df_sku = pd.DataFrame()
    
    return df_rich, df_sku

df_rich, df_sku = load_data()

# -------------------------------------------
# 3. サイドバー設定 (階層構造に変更)
# -------------------------------------------
st.sidebar.header("設定メニュー")

# 3-1. モード選択
app_mode = st.sidebar.selectbox(
    "モード選択",
    ["精度検証", "未来予測"]
)

# 3-2. シナリオ選択
target_scenario = st.sidebar.radio(
    "シナリオを選択",
    ("経営者向け: 売上予測", "物流担当: 注文数予測", "在庫担当: SKU需要予測")
)

# 3-3. モデル選択 (回帰 vs 時系列)
st.sidebar.markdown("---")
st.sidebar.subheader("モデル選択")

model_category = st.sidebar.selectbox(
    "モデルカテゴリ",
    ["回帰モデル", "時系列モデル"]
)

selected_model = ""
if model_category == "回帰モデル":
    selected_model = st.sidebar.selectbox("アルゴリズム", ["Random Forest"])
else:
    selected_model = st.sidebar.selectbox("アルゴリズム", ["Prophet", "ARIMA"])

# -------------------------------------------
# 4. データ準備 (シナリオ別)
# -------------------------------------------
target_col = ""
model_df = pd.DataFrame()
unit_label = ""

if target_scenario == "経営者向け: 売上予測":
    target_col = 'Sales'
    unit_label = "UKポンド"
    model_df = df_rich.copy()

elif target_scenario == "物流担当: 注文数予測":
    target_col = 'OrderCount'
    unit_label = "Orders"
    model_df = df_rich.copy()

elif target_scenario == "在庫担当: SKU需要予測":
    if df_sku.empty:
        st.error("Error: SKU data not found.")
        st.stop()
    
    sku_list = df_sku['StockCode'].unique()
    selected_sku = st.selectbox("Select SKU", sku_list)
    
    target_col = 'Quantity'
    unit_label = "Units"
    model_df = df_sku[df_sku['StockCode'] == selected_sku].copy()
    
    # 時系列モデルのために日付を完全連続化
    full_range = pd.date_range(start=model_df['Date'].min(), end=model_df['Date'].max(), freq='D')
    model_df = model_df.set_index('Date').reindex(full_range, fill_value=0).reset_index().rename(columns={'index': 'Date'})


# -------------------------------------------
# 5. データ分割 & 前処理 (モード別)
# -------------------------------------------
train_data = pd.DataFrame()
test_data = pd.DataFrame()
future_steps = 30 # 予測期間

# CSVアップロード用（Random Forestの未来予測用）
uploaded_future_df = pd.DataFrame()

if not model_df.empty:
    
    # --- A. Backtest Mode (過去データで検証) ---
    if app_mode == "精度検証":
        train_data = model_df.iloc[:-future_steps]
        test_data = model_df.iloc[-future_steps:]
    
    # --- B. Future Prediction (全データ学習 + 未来予測) ---
    else:
        train_data = model_df # 全量を学習
        
        # 予測期間をユーザーが設定可能に
        st.markdown("### 予測設定")
        future_steps = st.slider(
            "予測期間（日数）",
            min_value=7,
            max_value=365,
            value=30,
            step=1
        )
        
        # 回帰モデルの場合は、未来の特徴量(CSV)が必要
        if model_category == "回帰モデル":
            st.markdown("### ステップ1: 未来の特徴量データをアップロード")
            st.info("""
            **必須カラム**: `Date`, `Lag_1`, `Lag_7`, `DayOfWeek`
            
            **サンプルフォーマット**:
            | Date       | Lag_1 | Lag_7 | DayOfWeek |
            |------------|-------|-------|-----------|
            | 2025-12-01 | 1500  | 1400  | 0         |
            | 2025-12-02 | 1520  | 1420  | 1         |
            
            - `Lag_1`: 1日前の売上/注文数
            - `Lag_7`: 7日前の売上/注文数
            - `DayOfWeek`: 曜日（0=月曜 ～ 6=日曜）
            """, icon="📋")
            
            uploaded_file = st.file_uploader("CSV をアップロード", type="csv", key="future_csv")
            if uploaded_file:
                uploaded_future_df = pd.read_csv(uploaded_file)
                st.success(f"{len(uploaded_future_df)} 件のデータをアップロードしました")
                st.dataframe(uploaded_future_df.head(), use_container_width=True)
            else:
                st.error("エラー：Random Forest の未来予測には CSV ファイルが必須です")
                st.stop()
        
        # 時系列モデルは自動で未来の日付を生成
        else:
            st.markdown(f"### 向こう {future_steps} 日間を自動予測します")
            st.success(f"{selected_model} モデルで未来予測を実行中...")

# -------------------------------------------
# 6. モデル学習 & 予測実行
# -------------------------------------------
preds = []
y_true = None
dates = []

if not train_data.empty:
    with st.spinner(f'Training {selected_model}...'):
        
        # ==========================================
        # パターン1: Random Forest (回帰モデル)
        # ==========================================
        if selected_model == "Random Forest":
            # 特徴量作成 (Lag)
            df_rf = model_df.copy() # 全体から作る（Lag計算のため）
            df_rf['Lag_1'] = df_rf[target_col].shift(1)
            df_rf['Lag_7'] = df_rf[target_col].shift(7)
            df_rf['DayOfWeek'] = df_rf['Date'].dt.dayofweek
            df_rf = df_rf.dropna()
            
            features = ['Lag_1', 'Lag_7', 'DayOfWeek']
            
            # データの切り出し直し（dropna後）
            if app_mode == "精度検証":
                X_train = df_rf.iloc[:-future_steps][features]
                y_train = df_rf.iloc[:-future_steps][target_col]
                X_test = df_rf.iloc[-future_steps:][features]
                y_true = df_rf.iloc[-future_steps:][target_col]
                dates = df_rf.iloc[-future_steps:]['Date']
            else:
                # 未来予測
                X_train = df_rf[features]
                y_train = df_rf[target_col]
                X_test = uploaded_future_df[features] # アップロードされたCSVを使用
                dates = pd.to_datetime(uploaded_future_df['Date'])
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)

        # ==========================================
        # パターン2: Prophet (時系列モデル)
        # ==========================================
        elif selected_model == "Prophet":
            # Prophet専用の形式 (ds, y) に変換
            df_prophet = train_data[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
            
            m = Prophet()
            m.fit(df_prophet)
            
            if app_mode == "精度検証":
                # テスト期間の日付フレームを作成
                future = m.make_future_dataframe(periods=future_steps)
                forecast = m.predict(future)
                # 後ろから30日分を取得
                preds = forecast.iloc[-future_steps:]['yhat'].values
                y_true = test_data[target_col]
                dates = test_data['Date']
            else:
                # 本当の未来を作成
                future = m.make_future_dataframe(periods=future_steps)
                forecast = m.predict(future)
                preds = forecast.iloc[-future_steps:]['yhat'].values
                dates = forecast.iloc[-future_steps:]['ds']

        # ==========================================
        # パターン3: ARIMA (時系列モデル)
        # ==========================================
        elif selected_model == "ARIMA":
            # Series形式に変換
            train_series = train_data.set_index('Date')[target_col]
            
            # モデル構築 (パラメータは自動または固定 (5,1,0) など)
            arima_model = ARIMA(train_series, order=(5, 1, 0))
            arima_result = arima_model.fit()
            
            # 予測
            forecast_result = arima_result.forecast(steps=future_steps)
            preds = forecast_result.values
            
            if app_mode == "精度検証":
                y_true = test_data[target_col]
                dates = test_data['Date']
            else:
                # 未来の日付を生成
                last_date = train_data['Date'].max()
                dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)

# -------------------------------------------
# 7. 結果の可視化 (Visualization)
# -------------------------------------------
st.markdown("---")
st.subheader(":material/monitoring: 予測結果モニタリング")

fig = go.Figure()

# --- A. 実測値 (Backtestの場合のみ表示) ---
if y_true is not None:
    fig.add_trace(go.Scatter(
        x=dates, y=y_true, mode='lines', name='実績値 (Actual)',
        line=dict(color='gray', width=1.5), opacity=0.6
    ))

# --- B. シナリオ別の可視化設定 ---

# 経営層向け: 目標ラインの表示
if target_scenario == "経営者向け: 売上予測":
    # 予測値
    fig.add_trace(go.Scatter(
        x=dates, y=preds, mode='lines', name=f'AI予測 ({selected_model})',
        line=dict(color='#1E88E5', width=3)
    ))

    # 目標ライン（事実として表示）
    if y_true is not None:
        target_val = y_true.mean() * 1.05
    else:
        target_val = preds.mean() * 1.05
        
    fig.add_hline(
        y=target_val, 
        line_color="#D32F2F", line_width=2, line_dash="solid", # Material Red
        annotation_text=f"予算目標: {target_val:,.0f} {unit_label}", 
        annotation_position="top left"
    )

# 現場（物流・在庫）向け: 変動幅（バンド）の表示
else:
    
    # 変動リスク幅 (±20%と仮定)
    upper_band = preds * 1.2
    lower_band = preds * 0.8
    
    # バンド描画
    fig.add_trace(go.Scatter(
        x=dates, y=lower_band, mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=upper_band, mode='lines', line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(25, 118, 210, 0.1)', # Material Blue tint
        name='変動予測幅 (±20%)'
    ))
    
    # 中心予測値
    fig.add_trace(go.Scatter(
        x=dates, y=preds, mode='lines', name=f'AI予測 (中心値)',
        line=dict(color='#1E88E5', width=3)
    ))

# レイアウト設定
if app_mode == "精度検証":
    chart_title = f"検証結果: {selected_model}予測 vs 実績"
else:
    chart_title = f"未来予測: {selected_model}予測 による向こう30日間の推移"

fig.update_layout(
    title=chart_title,
    xaxis_title="日付",
    yaxis_title=unit_label,
    hovermode="x unified",
    template="plotly_white",
    margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------
# 8. 主要KPI (Key Performance Indicators)
# -------------------------------------------
st.subheader(":material/analytics: 主要指標 (Key Metrics)")

# カラム作成（事実データのみを表示）
col1, col2, col3 = st.columns(3)

# 共通指標: 期間平均値
avg_pred = preds.mean()
total_pred = preds.sum()

with col1:
    st.metric(
        label=f"期間平均 ({unit_label})", 
        value=f"{avg_pred:,.1f}"
    )

with col2:
    st.metric(
        label=f"期間合計 ({unit_label})", 
        value=f"{total_pred:,.1f}"
    )

# 精度指標 (Backtestモードのみ表示)
with col3:
    if app_mode == "精度検証" and y_true is not None:
        mask = y_true != 0
        if mask.sum() > 0:
            mape_val = np.mean(np.abs((y_true[mask] - preds[mask]) / y_true[mask])) * 100
            accuracy = max(0, 100 - mape_val)
        else:
            accuracy = 0
            mape_val = 0
            
        st.metric(
            label="モデル予測精度 (Accuracy)", 
            value=f"{accuracy:.1f}%",
            delta=f"誤差率(MAPE): {mape_val:.1f}%",
            delta_color="inverse" # 誤差が小さいほど良い色にする
        )
    else:
        # 未来予測モードの場合は、予測の信頼度に関する参考情報を表示
        st.info("※ 未来予測モードのため、正解データとの比較（精度算出）は行われません。", icon=":material/info:")

# -------------------------------------------
# 9. 詳細データテーブル (Data Grid)
# -------------------------------------------
with st.expander(":material/table_view: 詳細データを確認する"):
    # 結果をデータフレームにまとめる
    result_df = pd.DataFrame({
        "Date": dates,
        "Forecast": preds
    })
    
    if y_true is not None:
        result_df["Actual"] = y_true.values
        result_df["Diff"] = result_df["Forecast"] - result_df["Actual"]
    
    st.dataframe(result_df, use_container_width=True)

# -------------------------------------------
# 10. 採用モデルの技術解説 (Model Logic)
# -------------------------------------------
st.markdown("---")
with st.expander(":material/psychology: 採用アルゴリズムの技術的特徴 (Model Characteristics)", expanded=False):
    st.markdown(f"現在選択されている **{selected_model}** モデルのアルゴリズム概要と特性です。")

    # --- Random Forest ---
    if selected_model == "Random Forest":
        st.markdown("""
        #### :material/forest: Random Forest (ランダムフォレスト)
        * **分類**: 機械学習 / アンサンブル学習 (Bagging)
        * **仕組み**: 多数の「決定木」を構築し、各木の予測結果を平均して最終的な数値を算出します。
        * **特徴**:
            * データに含まれる**非線形な関係性**（単純な直線ではない複雑なパターン）を捉える能力が高いです。
            * 「曜日」「7日前の売上」といった特徴量の相互作用を学習します。
            * 外れ値やノイズに対して比較的頑健で、過学習しにくい特性があります。
        """)

    # --- Prophet ---
    elif selected_model == "Prophet":
        st.markdown("""
        #### :material/timeline: Prophet (プロフェット)
        * **分類**: 時系列解析 / 一般化加法モデル (GAM)
        * **仕組み**: Meta社によって開発されたモデル。「トレンド」「季節性」「休日効果」の3つを足し合わせることで予測します。
        * **特徴**:
            * **人間が解釈しやすい**構成要素（週周期、年周期など）でモデル化されます。
            * 欠損値（データの抜け）があっても補完せずにそのまま学習可能です。
            * トレンドの変化点（Change Points）を自動的に検知し、急激な需要変動に追従します。
        """)

    # --- ARIMA ---
    elif selected_model == "ARIMA":
        st.markdown("""
        #### :material/functions: ARIMA (自己回帰和分移動平均モデル)
        * **分類**: 統計的時系列モデル
        * **仕組み**: 以下の3つの要素を組み合わせて数式化します。
            1. **AR (AutoRegressive)**: 直近の過去データとの相関（自己相関）
            2. **I (Integrated)**: データの差分（トレンドの除去）
            3. **MA (Moving Average)**: 過去の予測誤差の影響
        * **特徴**:
            * データそのものが持つ「短期的な相関関係」を数理的にモデル化するため、**直近のトレンド延長**において高い精度を発揮しやすいです。
            * 季節変動が複雑な場合よりも、短期的な動きの慣性を捉えるのに適しています。
        """)
