# src/dashboard.py
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

try:
    from src.utils import MODELS_DIR
except ImportError:
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

st.set_page_config(
    layout="wide", 
    page_title="Blinkit Retention Dashboard", 
    initial_sidebar_state="expanded"
)

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
    
    numeric_cols = ['recency_days', 'frequency', 'monetary', 'delivery_ratio', 
                   'avg_order_value', 'negative_feedback_count', 'avg_rating', 'avg_sentiment']
    
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    if 'segment_name' not in df.columns:
        df['segment_name'] = 'Unsegmented'
    
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df

@st.cache_resource
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

df = load_feature_df()
models = load_models()

if df is None:
    st.title("Blinkit Retention Dashboard")
    st.error("No engineered feature CSV found in models/. Run train_segmentation.py first.")
    st.stop()

st.sidebar.header("Filters & Settings")

seg_options = ["All"] + sorted(df['segment_name'].unique().tolist())
selected_segment = st.sidebar.selectbox("Segment", seg_options, index=0)

min_freq_val = int(df['frequency'].min())
max_freq_val = int(df['frequency'].max())
if max_freq_val > min_freq_val:
    min_orders = st.sidebar.slider(
        "Min frequency (orders)", 
        min_freq_val, 
        max_freq_val, 
        min_freq_val
    )
else:
    min_orders = min_freq_val
    st.sidebar.info(f"All customers have frequency: {min_freq_val}")

date_as_of = st.sidebar.date_input("As-of date (for recency)", value=datetime.today().date())

filtered_df = df.copy()
if selected_segment != "All":
    filtered_df = filtered_df[filtered_df['segment_name'] == selected_segment]
filtered_df = filtered_df[filtered_df['frequency'] >= min_orders]

st.sidebar.info(f"Displaying {len(filtered_df):,} of {len(df):,} customers")

tabs = st.tabs(["Executive Summary", "Segmentation (3D)", "Churn Prediction"])

with tabs[0]:
    st.header("Executive Summary")
    
    if len(filtered_df) == 0:
        st.warning("No customers match the current filters.")
    else:
        customers_with_orders = filtered_df[filtered_df['frequency'] > 0]
        
        total_customers = len(filtered_df)
        
        if len(customers_with_orders) > 0:
            avg_order_value = customers_with_orders['avg_order_value'].mean()
            avg_delivery_ratio = customers_with_orders['delivery_ratio'].mean()
            avg_neg_feedback = filtered_df['negative_feedback_count'].mean()
        else:
            avg_order_value = 0
            avg_delivery_ratio = 0
            avg_neg_feedback = 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{total_customers:,}")
        with col2:
            st.metric("Avg Order Value", f"Rs. {avg_order_value:,.2f}")
        with col3:
            st.metric("Avg Delivery Ratio", f"{avg_delivery_ratio:.1%}")
        with col4:
            st.metric("Avg Negative Feedback", f"{avg_neg_feedback:.2f}")
        
        st.markdown("---")
        
        st.markdown("### Distribution of Key Features")
        c1, c2 = st.columns(2)
        
        with c1:
            if customers_with_orders['frequency'].nunique() > 1:
                fig = px.histogram(
                    customers_with_orders, x='frequency', nbins=30,
                    title="Order Frequency Distribution",
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("All customers have the same frequency.")
        
        with c2:
            if customers_with_orders['monetary'].nunique() > 1:
                fig2 = px.histogram(
                    customers_with_orders, x='monetary', nbins=40,
                    title="Monetary (Total Spend) Distribution",
                    color_discrete_sequence=['#764ba2']
                )
                fig2.update_layout(bargap=0.1)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("All customers have the same monetary value.")
        
        st.markdown("### Customer Segment Distribution")
        seg_dist = filtered_df['segment_name'].value_counts().reset_index()
        seg_dist.columns = ['Segment', 'Count']
        
        fig_seg = px.pie(
            seg_dist, values='Count', names='Segment',
            title="Customers by Segment",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_seg, use_container_width=True)

with tabs[1]:
    st.header("Segmentation (3D Interactive View)")
    
    if len(filtered_df) == 0:
        st.warning("No customers match the current filters.")
    else:
        viz_df = filtered_df[filtered_df['frequency'] > 0].copy()
        
        if len(viz_df) == 0:
            st.warning("No customers with orders in the current selection.")
        else:
            st.info(f"Showing {len(viz_df)} customers with at least one order")
            
            st.markdown("#### 3D Scatter: Frequency (x), Monetary (y), Delivery Ratio (z)")
            
            viz_df = viz_df[
                (viz_df['frequency'] > 0) & 
                (viz_df['monetary'] > 0) & 
                (viz_df['delivery_ratio'] >= 0)
            ].copy()
            
            if len(viz_df) < 3:
                st.warning(f"Not enough data points for 3D visualization (found {len(viz_df)})")
            else:
                viz_df['size_viz'] = np.log1p(viz_df['avg_order_value']) + 1
                
                fig_3d = px.scatter_3d(
                    viz_df,
                    x='frequency', 
                    y='monetary', 
                    z='delivery_ratio',
                    color='segment_name',
                    size='size_viz',
                    hover_name='customer_name',
                    hover_data={
                        'customer_id': True,
                        'customer_name': False,
                        'frequency': True,
                        'monetary': ':,.2f',
                        'delivery_ratio': ':.2%',
                        'avg_order_value': ':,.2f',
                        'negative_feedback_count': ':.0f',
                        'avg_rating': ':.2f',
                        'size_viz': False,
                        'segment_name': False
                    },
                    title=f"Customer Segments - {selected_segment} ({len(viz_df)} customers)",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    labels={
                        'frequency': 'Frequency (Orders)',
                        'monetary': 'Monetary (Rs.)',
                        'delivery_ratio': 'Delivery Ratio'
                    }
                )
                
                fig_3d.update_layout(
                    height=700,
                    scene=dict(
                        xaxis=dict(title="Frequency (Orders)", gridcolor="gray"),
                        yaxis=dict(title="Monetary (Rs.)", gridcolor="gray"),
                        zaxis=dict(title="Delivery Ratio", gridcolor="gray"),
                        bgcolor="black"
                    ),
                    paper_bgcolor="black",
                    plot_bgcolor="black",
                    font=dict(color="white"),
                    showlegend=True
                )
                
                fig_3d.update_traces(marker=dict(line=dict(width=0.5, color='white')))
                
                st.plotly_chart(fig_3d, use_container_width=True)
            
            st.markdown("#### Segment Profiles (Summary Statistics)")
            profile = viz_df.groupby('segment_name', observed=True).agg(
                count=('customer_id', 'nunique'),
                avg_freq=('frequency', 'mean'),
                avg_monetary=('monetary', 'mean'),
                avg_delivery_ratio=('delivery_ratio', 'mean'),
                avg_order_value=('avg_order_value', 'mean'),
                avg_neg_feedback=('negative_feedback_count', 'mean')
            ).reset_index().sort_values('avg_monetary', ascending=False)
            
            profile_display = profile.copy()
            profile_display['avg_freq'] = profile_display['avg_freq'].round(2)
            profile_display['avg_monetary'] = profile_display['avg_monetary'].apply(lambda x: f"Rs. {x:,.2f}")
            profile_display['avg_delivery_ratio'] = profile_display['avg_delivery_ratio'].apply(lambda x: f"{x:.1%}")
            profile_display['avg_order_value'] = profile_display['avg_order_value'].apply(lambda x: f"Rs. {x:,.2f}")
            profile_display['avg_neg_feedback'] = profile_display['avg_neg_feedback'].round(2)
            
            st.dataframe(profile_display, use_container_width=True, hide_index=True)
            
            st.markdown("#### Sample Customers from Selection")
            sample_cols = ['customer_id', 'customer_name', 'segment_name', 'frequency', 
                          'monetary', 'delivery_ratio', 'avg_order_value', 'avg_rating', 'negative_feedback_count']
            sample_df = viz_df[sample_cols].sort_values('monetary', ascending=False).head(100).copy()
            
            sample_df['monetary'] = sample_df['monetary'].apply(lambda x: f"Rs. {x:,.2f}")
            sample_df['delivery_ratio'] = sample_df['delivery_ratio'].apply(lambda x: f"{x:.1%}")
            sample_df['avg_order_value'] = sample_df['avg_order_value'].apply(lambda x: f"Rs. {x:,.2f}")
            
            st.dataframe(sample_df, use_container_width=True, hide_index=True)

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
            st.success("Churn model loaded successfully.")
            
            working_df = filtered_df.copy()
            
            for c in features_for_model:
                if c not in working_df.columns:
                    working_df[c] = 0
                working_df[c] = pd.to_numeric(working_df[c], errors='coerce').fillna(0)
            
            X = working_df[features_for_model]
            if hasattr(churn_model, 'predict_proba'):
                churn_probs = churn_model.predict_proba(X)[:, 1]
            else:
                churn_probs = churn_model.predict(X)
            
            working_df['churn_proba'] = churn_probs
            
            st.subheader("Top Customers at Risk")
            top_at_risk = working_df.sort_values('churn_proba', ascending=False).head(100)
            
            display_cols = ['customer_id', 'customer_name', 'segment_name', 'churn_proba',
                          'negative_feedback_count', 'avg_rating', 'frequency', 'monetary']
            top_at_risk_display = top_at_risk[display_cols].copy()
            top_at_risk_display['churn_proba'] = top_at_risk_display['churn_proba'].apply(lambda x: f"{x:.1%}")
            top_at_risk_display['monetary'] = top_at_risk_display['monetary'].apply(lambda x: f"Rs. {x:,.2f}")
            
            st.dataframe(top_at_risk_display.reset_index(drop=True), use_container_width=True)
            
            if hasattr(churn_model, 'feature_importances_'):
                st.subheader("Feature Importance")
                fi_df = pd.DataFrame({
                    'feature': features_for_model,
                    'importance': churn_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig_fi = px.bar(
                    fi_df, x='importance', y='feature', orientation='h',
                    title="Feature Importance (RandomForest)",
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            
            st.subheader("Churn Probability Distribution")
            fig_hist = px.histogram(
                working_df, x='churn_proba', nbins=30,
                title="Predicted Churn Probability",
                color_discrete_sequence=['#f87171']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("Churn model not found. Run train_churn.py first.")

st.sidebar.markdown("---")
st.sidebar.markdown("Blinkit Retention Dashboard")