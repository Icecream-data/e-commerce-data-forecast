import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# -------------------------------------------
# 1. ページ設定
# -------------------------------------------
st.set_page_config(page_title="EC需要予測AI", layout="wide")

st.title("AI需要予測ダッシュボード")
st.markdown("2010-12-08~2011-11-09のデータからトレンドを学習し、2011-11-10〜2011-12-09の30日間の数値を予測しました。")

# -------------------------------------------
# 2. データ読み込み
# -------------------------------------------
@st.cache_data
def load_data():
    # 全体データ
    df_rich = pd.read_csv('daily_sales_rich.csv')
    df_rich['Date'] = pd.to_datetime(df_rich['Date'])
    
    # 商品別データ
    try:
        df_sku = pd.read_csv('top_sku_daily.csv')
        df_sku['Date'] = pd.to_datetime(df_sku['Date'])
    except FileNotFoundError:
        df_sku = pd.DataFrame()
    
    return df_rich, df_sku

df_rich, df_sku = load_data()

# -------------------------------------------
# 3. サイドバー：シナリオ選択
# -------------------------------------------
st.sidebar.header("分析シナリオ")
scenario = st.sidebar.radio(
    "分析シナリオ別",
    ("経営層：売上予測", "物流担当：注文数予測", "在庫担当：商品別予測")
)

# -------------------------------------------
# 4. シナリオごとの設定
# -------------------------------------------
target_col = ""
model_df = pd.DataFrame()
unit_label = ""
graph_title = ""

if scenario == "経営層：売上予測":
    graph_title = "全社売上予測 vs 目標予算"
    target_col = 'Sales'
    unit_label = "UKポンド"
    model_df = df_rich.copy()

elif scenario == "物流担当：注文数予測":
    graph_title = "販売数予測と適正リソース帯"
    target_col = 'OrderCount'
    unit_label = "件"
    model_df = df_rich.copy()

elif scenario == "在庫担当：商品別予測":
    graph_title = "🛒 商品別販売予測と在庫リスク帯"
    if df_sku.empty:
        st.error("エラー: 'top_sku_daily.csv' がありません。")
        st.stop()
    
    sku_list = df_sku['StockCode'].unique()
    selected_sku = st.selectbox("予測したい商品コード (SKU)", sku_list)
    
    target_col = 'Quantity'
    unit_label = "個"
    model_df = df_sku[df_sku['StockCode'] == selected_sku].copy()
    full_range = pd.date_range(start=model_df['Date'].min(), end=model_df['Date'].max(), freq='D')
    model_df = model_df.set_index('Date').reindex(full_range, fill_value=0).reset_index().rename(columns={'index': 'Date'})

# -------------------------------------------
# 5. 機械学習モデルの構築と予測
# -------------------------------------------
if not model_df.empty:
    with st.spinner('AIが未来のパターンを計算中...'):
        # 特徴量作成
        model_df['Lag_1'] = model_df[target_col].shift(1)
        model_df['Lag_7'] = model_df[target_col].shift(7)
        model_df['DayOfWeek'] = model_df['Date'].dt.dayofweek
        df_ml = model_df.dropna()
        
        # データ分割
        test_days = 30
        train = df_ml.iloc[:-test_days]
        test = df_ml.iloc[-test_days:]
        
        # 学習と予測
        features = ['Lag_1', 'Lag_7', 'DayOfWeek']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train[features], train[target_col])
        preds = model.predict(test[features])

        # 精度評価（MAPE）
        mask = test[target_col] != 0
        y_true_safe = test.loc[mask, target_col]
        preds_safe = preds[mask]
        mape = np.mean(np.abs((y_true_safe - preds_safe) / y_true_safe)) * 100
        accuracy = max(0, 100 - mape)

        # -------------------------------------------
        # 6. 結果の可視化 (シナリオ別にリッチ化)
        # -------------------------------------------
        fig = go.Figure()

        # --- 共通: 実測値とAI予測値 ---
        fig.add_trace(go.Scatter(
            x=test['Date'], y=test[target_col], mode='lines', name='実測値',
            line=dict(color='gray', width=1.5), opacity=0.6
        ))
        fig.add_trace(go.Scatter(
            x=test['Date'], y=preds, mode='lines', name='AI予測値',
            line=dict(color='royalblue', width=3)
        ))

        # --- シナリオ別の装飾分岐 ---
        if scenario == "経営層：売上予測":
            # 目標ラインの追加
            # デモ用に「実測平均の5%アップ」を目標値と仮定
            target_value = test[target_col].mean() * 1.05
            
            fig.add_hline(y=target_value, line_dash="dash", line_color="firebrick", line_width=2,
                          annotation_text=f"目標予算: {target_value:,.0f} {unit_label}", 
                          annotation_position="top left")
            
            # 目標ラインの強調

        else: # 物流・在庫担当の場合
            # リスク帯（バンド）の表示
            # AI予測値の ±20% を「適正ゾーン」と定義
            upper_band = preds * 1.2
            lower_band = preds * 0.8
            
            # 下限の線（透明にする）
            fig.add_trace(go.Scatter(
                x=test['Date'], y=lower_band, mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ))
            # 上限の線（下限との間を塗りつぶす）
            fig.add_trace(go.Scatter(
                x=test['Date'], y=upper_band, mode='lines', line=dict(width=0),
                fill='tonexty', # ひとつ前のトレースとの間を塗る
                fillcolor='rgba(0, 200, 100, 0.2)', # 薄い緑色
                name='適正在庫/リソース帯 (予測±20%)'
            ))
            # ※この帯より上が「過剰」、下が「欠品/不足」リスクとなります

        # グラフのレイアウト調整
        fig.update_layout(title=graph_title, xaxis_title='日付', yaxis_title=unit_label, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------
        # 7. 評価とアクション
        # -------------------------------------------
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("予測精度 (Accuracy)", f"{accuracy:.1f}%")
            if accuracy >= 80:
                st.success("判定: 高精度 🌟")
            else:
                st.warning("判定: 改善の余地あり ⚠️")
        
        with col2:
            avg_val = preds.mean()
            if scenario == "経営層：売上予測":
                # 目標との比較アクション
                diff = avg_val - target_value
                if diff >= 0:
                     st.success(f"✅ 予測は目標を平均 **£{diff:,.0f}** 上回る見込みです。好調です！")
                else:
                     st.error(f"🚨 予測は目標を平均 **£{abs(diff):,.0f}** 下回る見込みです。対策が必要です。")

            elif scenario == "物流担当：注文数予測":
                staff_needed = int(upper_band.max() / 50) + 1 # バンドの上限に合わせて余裕を持つ
                st.info(f"繁忙日のピークに備え、最大 **{staff_needed}名**体制の準備を推奨します。")

            elif scenario == "在庫担当：商品別予測":
                # バンドの下限を安全在庫の基準にする
                safe_stock = int(lower_band.mean() * 7)
                st.warning(f"欠品リスク回避のため、最低 **{safe_stock}個** (適正帯下限の1週間分) の在庫維持を推奨します。")