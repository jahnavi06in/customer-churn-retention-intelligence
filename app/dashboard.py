import streamlit as st
import pandas as pd
import plotly.express as px

from src.data import load_data
from src.segmentation import evaluate_k, fit_segmentation
from src.churn_model import train_models, add_risk_scores

@st.cache_data
def cached_data():
    return load_data()

@st.cache_data
def cached_segmentation(df):
    evaluation = evaluate_k(df)
    # Choose a defensible k using the best score among 2-6, but expose the
    # evaluation rather than pretending the clusters are objectively "true".
    best_k = int(evaluation.loc[evaluation["silhouette_score"].idxmax(), "k"])
    segmented, profile, _, _ = fit_segmentation(df, k=best_k)
    return evaluation, best_k, segmented, profile

@st.cache_resource
def cached_models(df):
    fitted, metrics = train_models(df)
    best_name = metrics.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
    return fitted, metrics, best_name

@st.cache_data
def cached_risk(df, best_model_name, model_blob):
    # model_blob is intentionally used only as a cache key.
    # The actual model is passed separately by the caller.
    return df

def pct(x):
    return f"{x:.1f}%"

def render_dashboard():
    st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.8rem;}
    </style>
    """, unsafe_allow_html=True)

    st.title("Customer Churn & Retention Intelligence")
    st.caption("Segmentation • Churn Prediction • Customer Prioritization")

    try:
        df = cached_data()
    except Exception as e:
        st.error(str(e))
        st.info("Place the standard Churn_Modelling.csv dataset in data/ and restart the app.")
        return

    evaluation, best_k, segmented, profile = cached_segmentation(df)
    fitted_models, model_metrics, best_model_name = cached_models(df)
    best_model = fitted_models[best_model_name]

    risk_df = add_risk_scores(segmented, best_model)

    with st.sidebar:
        st.header("Filters")
        geographies = st.multiselect(
            "Geography", sorted(risk_df["Geography"].dropna().unique()),
            default=sorted(risk_df["Geography"].dropna().unique())
        )
        risk_levels = st.multiselect(
            "Risk Level", ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"]
        )
        active = st.multiselect(
            "Active Member", ["Active", "Inactive"],
            default=["Active", "Inactive"]
        )

    filtered = risk_df[
        risk_df["Geography"].isin(geographies)
        & risk_df["Risk_Level"].isin(risk_levels)
        & risk_df["IsActiveMember"].map({1: "Active", 0: "Inactive"}).isin(active)
    ].copy()

    tabs = st.tabs([
        "Executive Overview",
        "Customer Segmentation",
        "Churn Intelligence",
        "Retention Strategy",
        "Customer Explorer",
    ])

    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", f"{len(filtered):,}")
        c2.metric("Churn Rate", pct(filtered["Exited"].mean() * 100))
        c3.metric("High Risk", f"{(filtered['Risk_Level'] == 'High').sum():,}")
        c4.metric("Avg Churn Probability", pct(filtered["Churn_Probability"].mean() * 100))

        col1, col2 = st.columns(2)
        with col1:
            churn_geo = filtered.groupby("Geography", as_index=False)["Exited"].mean()
            churn_geo["Exited"] *= 100
            fig = px.bar(churn_geo, x="Geography", y="Exited",
                         title="Churn Rate by Geography",
                         labels={"Exited": "Churn Rate (%)"})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            risk_counts = filtered["Risk_Level"].value_counts().reindex(
                ["Low", "Medium", "High"], fill_value=0
            ).reset_index()
            risk_counts.columns = ["Risk_Level", "Customers"]
            fig = px.bar(risk_counts, x="Risk_Level", y="Customers",
                         title="Customer Risk Distribution")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Segmentation Validation")
        st.caption(
            "Silhouette score measures separation. Low scores mean these are "
            "analytical segments, not guaranteed natural customer types."
        )
        fig = px.line(
            evaluation, x="k", y="silhouette_score", markers=True,
            title="Silhouette Score by Number of Clusters",
            labels={"silhouette_score": "Silhouette Score", "k": "Clusters"}
        )
        fig.add_vline(x=best_k, line_dash="dash",
                      annotation_text=f"Selected k={best_k}")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Segment Profiles (k={best_k})")
        display = profile.copy()
        for col in ["Churn_Rate", "Active_Rate"]:
            display[col] = display[col].round(1)
        st.dataframe(display, use_container_width=True, hide_index=True)

        fig = px.scatter(
            risk_df, x="Age", y="Balance", color="Persona",
            size="Churn_Probability", hover_data=["Geography", "Gender", "Exited"],
            title="Customer Segments"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Churn Model Performance")
        m = model_metrics.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
            m[col] = m[col].round(3)
        st.dataframe(m, use_container_width=True, hide_index=True)

        st.success(f"Best model by ROC-AUC: {best_model_name}")

        geo = filtered.groupby("Geography", as_index=False)["Churn_Probability"].mean()
        geo["Churn_Probability"] *= 100
        fig = px.bar(
            geo, x="Geography", y="Churn_Probability",
            title="Predicted Churn Risk by Geography",
            labels={"Churn_Probability": "Average Predicted Risk (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("Retention Prioritization")
        priority = filtered.sort_values(
            "Retention_Priority", ascending=False
        ).head(20).copy()

        st.metric("High-risk customers", f"{(filtered['Risk_Level'] == 'High').sum():,}")
        st.caption("Priority combines predicted churn risk with balance exposure.")

        cols = [
            "CustomerId", "Geography", "Age", "Balance",
            "Churn_Probability", "Risk_Level", "Retention_Priority",
            "Recommended_Action"
        ]
        priority["Churn_Probability"] = priority["Churn_Probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(priority[cols], use_container_width=True, hide_index=True)

    with tabs[4]:
        st.subheader("Customer Explorer")
        cols = [
            "CustomerId", "Surname", "Geography", "Gender", "Age",
            "Balance", "NumOfProducts", "IsActiveMember",
            "Persona", "Churn_Probability", "Risk_Level",
            "Recommended_Action"
        ]
        explorer = filtered[cols].copy()
        explorer["Churn_Probability"] = explorer["Churn_Probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(explorer, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "Analytical note: K-Means segments are exploratory and should not be "
        "interpreted as causal customer types. Churn probabilities are model "
        "scores, not guarantees of customer behavior."
    )
