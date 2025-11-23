import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------
# 1. ページ設定
# -------------------------------------------
st.set_page_config(page_title="EC需要予測AI", layout="wide")

st.title("AI需要予測ダッシュボード")
st.markdown("過去のデータからトレンドを学習し、**向こう1ヶ月の数値**を予測します。")

# -------------------------------------------
# 2. データ読み込み
# -------------------------------------------
@st.cache_data
def load_data():
    # 全体データ（リッチ版）
    df_rich = pd.read_csv('daily_sales_rich.csv')
    df_rich['Date'] = pd.to_datetime(df_rich['Date'])
    
    # 商品別データ（もしファイルがなければ空のDataFrameを返す）
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
    "誰のために予測しますか？",
    ("経営層：売上予測", "物流担当：注文数予測", "在庫担当：商品別予測")
)

# -------------------------------------------
# 4. シナリオごとの設定
# -------------------------------------------
target_col = ""
model_df = pd.DataFrame()
unit_label = ""

if scenario == "経営層：売上予測":
    st.subheader("💰 全社売上予測 (Sales Forecast)")
    target_col = 'Sales'
    unit_label = "UKポンド"
    model_df = df_rich.copy()

elif scenario == "物流担当：注文数予測":
    st.subheader("📦 出荷・注文数予測 (Order Count)")
    target_col = 'OrderCount'
    unit_label = "件"
    model_df = df_rich.copy()

elif scenario == "在庫担当：商品別予測":
    st.subheader("🛒 商品別 販売個数予測")
    
    # データファイルのチェック
    if df_sku.empty:
        st.error("エラー: 'top_sku_daily.csv' が見つかりません。")
        st.stop()
    
    # 商品選択
    sku_list = df_sku['StockCode'].unique()
    selected_sku = st.selectbox("予測したい商品コード (SKU)", sku_list)
    
    target_col = 'Quantity'
    unit_label = "個"
    
    # その商品のデータだけ抽出
    model_df = df_sku[df_sku['StockCode'] == selected_sku].copy()
    
    # 日付の歯抜けを埋める（商品別は飛び飛びになりがちなので必須）
    full_range = pd.date_range(start=model_df['Date'].min(), end=model_df['Date'].max(), freq='D')
    model_df = model_df.set_index('Date').reindex(full_range, fill_value=0).reset_index().rename(columns={'index': 'Date'})

# -------------------------------------------
# 5. 機械学習モデルの構築と予測
# -------------------------------------------
if not model_df.empty:
    # ローディング表示
    with st.spinner('AIが過去のパターンを学習中...'):
        
        # 特徴量エンジニアリング（AIへのヒント作成）
        # 1日前、7日前、曜日などを教える
        model_df['Lag_1'] = model_df[target_col].shift(1)
        model_df['Lag_7'] = model_df[target_col].shift(7)
        model_df['DayOfWeek'] = model_df['Date'].dt.dayofweek
        
        # NaN（欠損）を除去
        df_ml = model_df.dropna()
        
        # データの分割（ラスト30日をテスト＝未来と見立てる）
        test_days = 30
        train = df_ml.iloc[:-test_days]
        test = df_ml.iloc[-test_days:]

        # =======================================================
        # データの中身をデバッグ表示する機能
        # =======================================================
        with st.expander("🔍 学習・テストデータの中身を確認する (Debug)"):
            st.markdown("### 1. 学習データ (Train)")
            st.info(f"期間: {train['Date'].min().date()} 〜 {train['Date'].max().date()} | 件数: {len(train)}件")
            st.dataframe(train.tail(5)) # 末尾5行を表示
            st.caption("☝️ 学習データの最後（テスト直前）です。Lag_1などが正しく入っているか確認。")

            st.markdown("### 2. テストデータ (Test)")
            st.info(f"期間: {test['Date'].min().date()} 〜 {test['Date'].max().date()} | 件数: {len(test)}件")
            st.dataframe(test.head(5)) # 先頭5行を表示
            st.caption("☝️ ここが予測対象です。実測値(Sales/Order/Quantity)を隠して、Lag情報から当てに行きます。")
        
        # 学習に使う項目
        features = ['Lag_1', 'Lag_7', 'DayOfWeek']
        
        # モデル作成（ランダムフォレスト）
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train[features], train[target_col])
        
        # 予測実行
        preds = model.predict(test[features])

        # -------------------------------------------
        # 📊 精度評価 (MAPE & Accuracy)
        # -------------------------------------------
        # 1. ゼロ除算を防ぐため、実測値が0の行を除外して計算する
        mask = test[target_col] != 0
        y_true_safe = test.loc[mask, target_col]
        preds_safe = preds[mask]

        # 2. MAPE計算 ( |実測 - 予測| / 実測 ) の平均
        mape = np.mean(np.abs((y_true_safe - preds_safe) / y_true_safe)) * 100

        # 3. 精度 (Accuracy) として表示 (100% - MAPE)
        # ※MAPEが100%を超えるとマイナスになるので0%で下限クリップ
        accuracy = max(0, 100 - mape)

        # -------------------------------------------
        # 画面表示
        # -------------------------------------------
        st.markdown("### モデルの予測精度")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("予測精度 (Accuracy)", f"{accuracy:.1f}%")
            st.caption("100% - MAPE で算出")

        with col2:
            st.metric("平均誤差率 (MAPE)", f"{mape:.1f}%")
            st.caption("実測値に対して平均何％ズレたか")

        with col3:
            # 評価コメントの自動判定
            if mape < 10:
                st.success("判定: 非常に高精度 🌟")
            elif mape < 20:
                st.info("判定: 良好 ✅")
            elif mape < 30:
                st.warning("判定: 普通 (改善余地あり) ⚠️")
            else:
                st.error("判定: 精度低 (要チューニング) 🚨")

        st.markdown("---")
        
        # -------------------------------------------
        # 6. 結果の可視化
        # -------------------------------------------
        # グラフ作成
        fig = go.Figure()
        
        # 実測値（青線）
        fig.add_trace(go.Scatter(
            x=test['Date'], y=test[target_col], 
            mode='lines', name='実測値', 
            line=dict(color='royalblue', width=2)
        ))
        
        # AI予測値（赤の点線）
        fig.add_trace(go.Scatter(
            x=test['Date'], y=preds, 
            mode='lines', name='AI予測値', 
            line=dict(color='firebrick', width=2, dash='dot')
        ))
        
        fig.update_layout(title=f'向こう30日間の予測比較', xaxis_title='日付', yaxis_title=unit_label)
        st.plotly_chart(fig, use_container_width=True)
        
        # -------------------------------------------
        # 7. ビジネスアクションの提案
        # -------------------------------------------
        st.markdown("---")
        st.subheader("💡 AIからのアクション提案")
        
        col1, col2 = st.columns(2)
        
        avg_val = preds.mean()
        
        with col1:
            st.metric(f"予測期間の平均値", f"{avg_val:,.1f} {unit_label}")
        
        with col2:
            if scenario == "物流担当：注文数予測":
                # 例えば1日あたり、1人50件処理できるとする
                staff_needed = int(avg_val / 50) + 1
                st.info(f"推奨人員: 1人50件処理できるとすると、1日あたり約 **{staff_needed}名** のスタッフ配置が必要です。")
            
            elif scenario == "在庫担当：商品別予測":
                # 安全在庫（1週間分）
                safe_stock = int(avg_val * 7)
                st.warning(f"推奨在庫: 1週間分の安全在庫を持つとすると、欠品を防ぐため最低 **{safe_stock}個** の在庫確保を推奨します。")
            
            else:
                st.success("予算管理: 予測値に基づいて、来月のキャッシュフロー計画を調整してください。")