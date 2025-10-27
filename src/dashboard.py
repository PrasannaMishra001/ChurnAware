import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from src.utils import MODELS_DIR

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    layout="wide",
    page_title="Blinkit Retention Dashboard",
    initial_sidebar_state="expanded",
)

# -------------------------
# Custom CSS for Blinkit Theme
# -------------------------
st.markdown("""
    <style>
    :root {
        --blinkit-yellow: #F8CB46;
        --blinkit-green: #0C831F;
        --blinkit-black: #000000;
    }
    html, body, [class*="css"] {
        color: var(--blinkit-black);
        font-family: 'Inter', sans-serif;
    }
    .stMetric label, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--blinkit-green);
    }
    .stMetric-value {
        color: var(--blinkit-yellow) !important;
    }
    .stDataFrame, .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 0 8px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_feature_df():
    p_with = os.path.join(MODELS_DIR, "customer_feature_engineered_with_segments.csv")
    p = os.path.join(MODELS_DIR, "customer_feature_engineered.csv")
    if os.path.exists(p_with):
        df = pd.read_csv(p_with)
    elif os.path.exists(p):
        df = pd.read_csv(p)
    else:
        return None

    numeric_cols = [
        'recency_days', 'frequency', 'monetary', 'delivery_ratio',
        'avg_order_value', 'negative_feedback_count', 'avg_rating', 'avg_sentiment'
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0

    if 'segment_name' not in df.columns:
        df['segment_name'] = 'Unsegmented'
    return df


@st.cache_data
def load_models():
    models = {}
    gmm_path = os.path.join(MODELS_DIR, "customer_segments_gmm.pkl")
    churn_path = os.path.join(MODELS_DIR, "churn_prediction_model.pkl")
    try:
        if os.path.exists(gmm_path):
            models['gmm'] = joblib.load(gmm_path)
    except Exception as e:
        st.warning(f"Failed to load GMM model: {e}")
    try:
        if os.path.exists(churn_path):
            models['churn'] = joblib.load(churn_path)
    except Exception as e:
        st.warning(f"Failed to load churn model: {e}")
    return models


def compute_churn_probs(model, df, feature_cols):
    X = df[feature_cols].fillna(0)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[:, 1]
        except Exception:
            proba = model.predict(X)
    else:
        proba = model.predict(X)
    return np.array(proba)


def feature_importances_table(model, feature_cols):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        return pd.DataFrame({
            "feature": feature_cols,
            "importance": imp
        }).sort_values("importance", ascending=False)
    return None

# -------------------------
# Load Data & Models
# -------------------------
df = load_feature_df()
models = load_models()

if df is None:
    st.title("Blinkit — Sentiment-aware R+FMD Retention Dashboard")
    st.error("No engineered feature CSV found in models/. Run train_segmentation.py first.")
    st.stop()

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("Filters & Settings")
seg_options = ["All"] + sorted(df['segment_name'].unique().tolist())
selected_segment = st.sidebar.selectbox("Segment", seg_options, index=0)
min_orders = st.sidebar.slider(
    "Min frequency (orders)",
    int(df['frequency'].min()), int(df['frequency'].max()), int(df['frequency'].min())
)
date_as_of = st.sidebar.date_input("As-of date (for recency)", value=datetime.today().date())

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Executive Summary", "Segmentation (3D)", "Churn Prediction & At-Risk"])

# -------------------------
# Executive Summary
# -------------------------
with tabs[0]:
    st.header("Executive Summary")
    total_customers = int(df['customer_id'].nunique())
    avg_order_value = df['avg_order_value'].replace([np.inf, -np.inf], np.nan).fillna(0).mean()
    avg_delivery_ratio = df['delivery_ratio'].mean()
    avg_neg_feedback = df['negative_feedback_count'].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{total_customers:,}")
    k2.metric("Avg Order Value", f"₹ {avg_order_value:,.2f}")
    k3.metric("Avg Delivery Ratio", f"{avg_delivery_ratio:.2f}")
    k4.metric("Avg Negative Feedback", f"{avg_neg_feedback:.2f}")

    st.markdown("### Distribution of Key Features")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='frequency', nbins=30, title="Order Frequency Distribution")
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    with c2:
        fig2 = px.histogram(df, x='monetary', nbins=40, title="Monetary (Total Spend) Distribution")
        st.plotly_chart(fig2, width='stretch', config={'displayModeBar': False})

    st.markdown("### Top Feedback Categories")
    if 'feedback_category' in df.columns:
        cat_counts = df['feedback_category'].value_counts().reset_index()
        cat_counts.columns = ['category', 'count']
        fig3 = px.bar(cat_counts, x='category', y='count', title="Feedback Categories")
        st.plotly_chart(fig3, width='stretch', config={'displayModeBar': False})
    else:
        st.info("Feedback categories not available in engineered dataset.")

# -------------------------
# Segmentation (3D)
# -------------------------
with tabs[1]:
    st.header("Segmentation (3D Interactive View)")
    seg_df = df.copy()
    if selected_segment != "All":
        seg_df = seg_df[seg_df['segment_name'] == selected_segment]
    seg_df = seg_df[seg_df['frequency'] >= min_orders]

    if len(seg_df) == 0:
        st.warning("No customers match the current filters.")
    else:
        st.markdown("#### 3D Scatter: Frequency (x), Monetary (y), Delivery Ratio (z)")
        fig_3d = px.scatter_3d(
            seg_df,
            x='frequency',
            y='monetary',
            z='delivery_ratio',
            color='segment_name',
            size='avg_order_value',
            hover_data=['customer_id', 'customer_name', 'negative_feedback_count', 'avg_rating'],
            title=f"Customer Segments — {selected_segment}"
        )
        fig_3d.update_layout(
            height=700,
            scene=dict(
                xaxis_title="Frequency",
                yaxis_title="Monetary",
                zaxis_title="Delivery Ratio"
            ),
            margin=dict(l=0, r=0, b=0, t=50),
        )
        st.plotly_chart(fig_3d, width='stretch', config={'displayModeBar': True})

        st.markdown("#### Segment Profiles (Summary Stats)")
        profile = seg_df.groupby('segment_name').agg(
            count=('customer_id', 'nunique'),
            avg_freq=('frequency', 'mean'),
            avg_monetary=('monetary', 'mean'),
            avg_delivery_ratio=('delivery_ratio', 'mean'),
            avg_neg_feedback=('negative_feedback_count', 'mean')
        ).reset_index().sort_values('avg_monetary', ascending=False)
        st.dataframe(profile, width='stretch')

        st.markdown("#### Sample Customers from Selection")
        st.dataframe(
            seg_df[['customer_id', 'customer_name', 'segment_name', 'frequency',
                    'monetary', 'delivery_ratio', 'avg_rating', 'negative_feedback_count']]
            .sort_values('monetary', ascending=False)
            .head(100),
            width='stretch'
        )

# -------------------------
# Churn Prediction & At-Risk
# -------------------------
with tabs[2]:
    st.header("Churn Prediction & At-Risk Customers")
    churn_model_path = os.path.join(MODELS_DIR, "churn_prediction_model.pkl")
    features_for_model = [
        'recency_days', 'frequency', 'monetary', 'delivery_ratio',
        'avg_rating', 'negative_feedback_count', 'avg_sentiment', 'avg_order_value'
    ]

    if os.path.exists(churn_model_path):
        try:
            churn_model = joblib.load(churn_model_path)
            st.success("✅ Churn model loaded successfully.")
            working_df = df.copy()
            if selected_segment != "All":
                working_df = working_df[working_df['segment_name'] == selected_segment]
            working_df = working_df[working_df['frequency'] >= min_orders]

            for c in features_for_model:
                if c not in working_df.columns:
                    working_df[c] = 0

            churn_probs = compute_churn_probs(churn_model, working_df, features_for_model)
            working_df = working_df.assign(churn_proba=churn_probs)

            st.subheader("Top Customers at Risk")
            top_at_risk = working_df.sort_values('churn_proba', ascending=False).head(100)
            st.dataframe(
                top_at_risk[['customer_id', 'customer_name', 'segment_name', 'churn_proba',
                             'negative_feedback_count', 'avg_rating', 'frequency', 'monetary']],
                width='stretch'
            )

            fi = feature_importances_table(churn_model, features_for_model)
            if fi is not None:
                st.subheader("Feature Importance (Churn Model)")
                fig_fi = px.bar(fi, x='importance', y='feature', orientation='h', title="Feature Importance (RandomForest)")
                st.plotly_chart(fig_fi, width='stretch', config={'displayModeBar': False})
            else:
                st.info("Model does not expose feature importances.")

            st.subheader("Churn Probability Distribution")
            fig_hist = px.histogram(working_df, x='churn_proba', nbins=30, title="Predicted Churn Probability")
            st.plotly_chart(fig_hist, width='stretch', config={'displayModeBar': False})

            st.markdown("### Single Customer Prediction")
            cid = st.text_input("Enter customer_id to predict", value="")
            if cid:
                try:
                    cid_int = int(cid)
                    row = df[df['customer_id'] == cid_int]
                    if row.empty:
                        st.warning("customer_id not found in dataset.")
                    else:
                        X_row = row[features_for_model].fillna(0)
                        if hasattr(churn_model, "predict_proba"):
                            proba = churn_model.predict_proba(X_row)[0, 1]
                        else:
                            proba = churn_model.predict(X_row)[0]
                        st.write(row.T)
                        st.success(f"Predicted churn probability: {proba:.3f}")
                except Exception as e:
                    st.error(f"Error parsing customer_id: {e}")
        except Exception as e:
            st.error(f"Failed to load churn model: {e}")
    else:
        st.info("Churn model not found. Run `python -m src.train_churn` to generate it.")

    st.markdown("---")
    st.markdown("**Notes:** Churn model expects features: " + ", ".join(features_for_model))
