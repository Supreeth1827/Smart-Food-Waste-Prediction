import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt

st.set_page_config(page_title="Smart Canteen Food Waste Optimizer", page_icon="🍽️", layout="wide")

# ---- Load Saved Artifacts ----
@st.cache_resource
def load_bundle():
    return joblib.load("canteen_outputs/smart_food_waste_model.pkl")

@st.cache_data
def load_data():
    clean_df = pd.read_csv("canteen_outputs/canteen_cleaned_dataset.csv", parse_dates=["date"])
    pred_df = pd.read_csv("canteen_outputs/prediction_results.csv", parse_dates=["date"])
    comp_df = pd.read_csv("canteen_outputs/model_comparison.csv")
    return clean_df, pred_df, comp_df

bundle = load_bundle()
clean_df, pred_df, comp_df = load_data()

MENU = bundle["menu"]
ALL_FEATURES = bundle["all_features"]
final_model = bundle["model"]
oof_rel_errors = bundle["oof_relative_errors"]

# ---- Helper Functions (Matching Notebook Logic) ----
def get_academic_period(date):
    d = pd.Timestamp(date)
    if (d.month == 5 and d.day >= 15) or d.month == 6:
        return "Vacation"
    if (d.month == 12 and d.day >= 22) or (d.month == 1 and d.day <= 4):
        return "Vacation"
    if (d.month == 11 and d.day >= 15) or (d.month == 4 and d.day >= 15):
        return "Exam"
    return "Regular"

def compute_buffer_for_level(category, service_level):
    errs = oof_rel_errors.get(category, np.concatenate(list(oof_rel_errors.values())))
    q = np.quantile(errs, service_level)
    return float(np.clip(q, 0.0, 0.60))

def build_feature_row(attendance, menu_item, weather, temperature, date, is_holiday):
    date = pd.Timestamp(date).normalize()
    category, _, price = MENU[menu_item]
    hist = clean_df[(clean_df["menu_item"] == menu_item) & (clean_df["date"] < date)].sort_values("date")
    if hist.empty:
        hist = clean_df[clean_df["menu_item"] == menu_item].sort_values("date").head(7)
    last7, last3 = hist.tail(7), hist.tail(3)
    rate = last7["food_demand"].sum() / max(last7["attendance"].sum(), 1)

    row = {
        "date": date,
        "attendance": float(attendance),
        "temperature": float(temperature),
        "is_holiday": int(bool(is_holiday)),
        "price": price,
        "historical_avg_demand": last7["food_demand"].mean(),
        "demand_lag_1": float(last7["food_demand"].iloc[-1]),
        "hist_avg_3": last3["food_demand"].mean(),
        "hist_demand_per_student": rate,
        "expected_from_history": rate * float(attendance),
        "menu_item": menu_item,
        "food_category": category,
        "weather": weather,
        "academic_period": get_academic_period(date),
    }
    frame = pd.DataFrame([row])
    d = pd.to_datetime(frame["date"])
    frame["day_of_week"] = d.dt.day_name()
    frame["month"] = d.dt.month
    frame["day_of_month"] = d.dt.day
    frame["week_of_year"] = d.dt.isocalendar().week.astype(int)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)
    frame["is_saturday"] = (d.dt.dayofweek == 5).astype(int)
    return frame[ALL_FEATURES]

# ---- Header ----
st.title("🍽️ Smart Canteen Food Waste Prediction & Optimization")
st.caption(f"Selected Production Model: **{bundle['best_model_name']}** | Data Source: `{bundle['data_source']}` (Seed {bundle['seed']})")

tab1, tab2, tab3 = st.tabs(["🔮 Live Kitchen Predictor", "♻️ Waste & Cost Optimization (Test Set)", "📊 Model Performance & EDA"])

# ==================== TAB 1: LIVE PREDICTOR ====================
with tab1:
    col_in, col_out = st.columns([1, 1.4], gap="large")
    with col_in:
        st.subheader("Morning Kitchen Inputs")
        in_date = st.date_input("Serving Date", value=dt.date(2025, 3, 12))
        in_item = st.selectbox("Menu Item", options=list(MENU.keys()), index=3)
        in_att = st.number_input("Expected Student Attendance", min_value=10, max_value=6000, value=2200, step=50)
        in_wx = st.selectbox("Weather Forecast", options=["Sunny", "Cloudy", "Rainy"], index=0)
        in_temp = st.slider("Temperature (°C)", min_value=10.0, max_value=45.0, value=30.0, step=0.5)
        in_hol = st.checkbox("National / Festival Holiday", value=False)
        in_sl = st.slider("Target Service Level (No-Shortage Guarantee)", min_value=0.70, max_value=0.99, value=0.90, step=0.01)

    with col_out:
        st.subheader("Recommended Preparation Plan")
        pdate = pd.Timestamp(in_date)
        if pdate.dayofweek == 6:
            st.warning("⚠️ Selected date is a Sunday (canteen closed in training data).")

        x_row = build_feature_row(in_att, in_item, in_wx, in_temp, in_date, in_hol)
        pred_demand = float(max(final_model.predict(x_row)[0], 0))
        cat = MENU[in_item][0]
        buf = compute_buffer_for_level(cat, in_sl)
        rec_prep = int(np.ceil(pred_demand + buf * max(pred_demand, 20)))

        # Uncertainty simulation from out-of-fold relative errors
        rel = oof_rel_errors.get(cat, np.array([0.0]))
        sims = np.maximum(pred_demand + rel * max(pred_demand, 20), 0)
        exp_left = float(np.mean(np.maximum(rec_prep - sims, 0)))
        shortage_risk = 100 * float(np.mean(sims > rec_prep))

        # Traditional rule comparison
        hist7 = float(x_row["historical_avg_demand"].iloc[0])
        trad_prep = int(np.ceil(hist7 * 1.20 * (0.4 if in_hol else 1.0) * (0.7 if pdate.dayofweek == 5 else 1.0)))

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Demand", f"{pred_demand:.0f} portions")
        m2.metric("Recommended Cook Qty", f"{rec_prep} portions", delta=f"+{buf:.1%} safety buffer")
        m3.metric("Traditional Rule Qty", f"{trad_prep} portions", delta=f"{rec_prep - trad_prep:+d} vs Traditional", delta_color="inverse")

        m4, m5, m6 = st.columns(3)
        m4.metric("Academic Period", get_academic_period(pdate))
        m5.metric("Expected Leftover", f"{exp_left:.1f} portions")
        m6.metric("Shortage Risk", f"{shortage_risk:.1f}%")

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=sims, nbinsx=25, name="Simulated Demand Outcomes", marker_color="#8fb1d9"))
        fig_dist.add_vline(x=pred_demand, line_color="black", line_width=2, annotation_text="Forecast")
        fig_dist.add_vline(x=rec_prep, line_color="#2ca02c", line_dash="dash", line_width=2, annotation_text="ML Prep")
        fig_dist.add_vline(x=trad_prep, line_color="#d62728", line_dash="dot", line_width=2, annotation_text="Traditional")
        fig_dist.update_layout(title=f"Demand Uncertainty Distribution — {in_item} ({cat})", xaxis_title="Portions", height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig_dist, use_container_width=True)

# ==================== TAB 2: WASTE OPTIMIZATION ====================
with tab2:
    st.subheader("Unseen Test Period Simulation (282 Days)")
    trad_waste = pred_df["traditional_leftover"].sum()
    ml_waste = pred_df["ml_leftover"].sum()
    saved_portions = trad_waste - ml_waste
    saved_pct = 100 * saved_portions / trad_waste

    trad_unmet = 100 * pred_df["traditional_shortage"].sum() / pred_df["actual_demand"].sum()
    ml_unmet = 100 * pred_df["ml_shortage"].sum() / pred_df["actual_demand"].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Traditional Leftover", f"{trad_waste:,.0f} portions")
    k2.metric("ML-Based Leftover", f"{ml_waste:,.0f} portions", delta=f"-{saved_pct:.1f}% waste", delta_color="normal")
    k3.metric("Traditional Unmet Demand", f"{trad_unmet:.2f}%")
    k4.metric("ML Unmet Demand", f"{ml_unmet:.2f}%", delta=f"{ml_unmet - trad_unmet:.2f} pts", delta_color="inverse")

    c1, c2 = st.columns(2)
    with c1:
        cat_waste = pred_df.groupby("food_category")[["traditional_leftover", "ml_leftover"]].sum().reset_index()
        cat_melt = cat_waste.melt(id_vars="food_category", var_name="Policy", value_name="Leftover Portions")
        fig_cat = px.bar(cat_melt, x="food_category", y="Leftover Portions", color="Policy", barmode="group",
                         title="Leftover Portions by Category", color_discrete_sequence=["#d62728", "#2ca02c"])
        st.plotly_chart(fig_cat, use_container_width=True)

    with c2:
        pred_df["month"] = pred_df["date"].dt.to_period("M").astype(str)
        m_waste = pred_df.groupby("month")[["traditional_leftover", "ml_leftover"]].sum().reset_index()
        fig_mo = px.line(m_waste, x="month", y=["traditional_leftover", "ml_leftover"], markers=True,
                         title="Monthly Leftover Trend (Notice Vacation Spikes)", color_discrete_sequence=["#d62728", "#2ca02c"])
        st.plotly_chart(fig_mo, use_container_width=True)

# ==================== TAB 3: MODEL COMPARISON & EDA ====================
with tab3:
    st.subheader("5-Fold TimeSeriesSplit & Unseen Test Metrics")
    st.dataframe(comp_df.style.highlight_min(subset=["CV_RMSE", "Test_RMSE", "Test_MAE"], color="#d4edda"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig_rmse = px.bar(comp_df, x="Model", y=["CV_RMSE", "Test_RMSE"], barmode="group", title="Validation vs Test RMSE (Lower is Better)")
        st.plotly_chart(fig_rmse, use_container_width=True)
    with c4:
        fig_scat = px.scatter(pred_df, x="actual_demand", y="best_model_prediction", color="food_category", opacity=0.6,
                              title=f"Actual vs Predicted Demand ({bundle['best_model_name']})")
        max_v = pred_df["actual_demand"].max()
        fig_scat.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v, line=dict(dash="dash", color="black"))
        st.plotly_chart(fig_scat, use_container_width=True)
