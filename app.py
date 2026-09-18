import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import subprocess
import sys
import streamlit.components.v1 as components
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
from sentence_transformers import SentenceTransformer
@st.cache_resource
def load_intent_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

intent_model = load_intent_model()
def speak_text(text, language):

    language_codes = {
        "English": "en",
        "ಕನ್ನಡ (Kannada)": "kn",
        "हिन्दी (Hindi)": "hi"
    }

    language_code = language_codes.get(language, "en")

    tts = gTTS(
        text=text,
        lang=language_code
    )

    audio_file = "voice_response.mp3"

    tts.save(audio_file)

    st.audio(
        audio_file,
        format="audio/mp3",
        autoplay=True
)
st.set_page_config(
    page_title="Retail Sales Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ==============================
# 📥 Upload Sales Dataset
# ==============================

def load_default_sales_data():
    return pd.read_csv("data/processed/featured_sales.csv")


DEFAULT_DATA_PATH = "data/processed/featured_sales.csv"

with st.sidebar:
    st.divider()
    st.markdown("### 📥 Upload New Sales Dataset")
    st.caption(
        "Upload a CSV to analyze a new retail sales dataset. "
        "The existing dataset is used when no file is uploaded."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="sales_dataset_uploader"
    )

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)

        required_columns = [
            "Sales", "Profit", "Quantity", "Order ID",
            "Category", "Region", "Segment", "Order Date"
        ]

        missing_columns = [
            column for column in required_columns
            if column not in uploaded_df.columns
        ]

        if missing_columns:
            st.sidebar.error(
                "❌ Uploaded file is missing required columns: "
                + ", ".join(missing_columns)
            )
            df = load_default_sales_data()
            st.sidebar.info("Using the original dataset instead.")
        else:
            df = uploaded_df.copy()

            # Create optional columns automatically when missing.
            df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
            if "Year" not in df.columns:
                df["Year"] = df["Order Date"].dt.year
            if "Month" not in df.columns:
                df["Month"] = df["Order Date"].dt.month
            if "Discount" not in df.columns:
                df["Discount"] = 0.0

            st.sidebar.success(
                f"✅ Uploaded dataset loaded: {len(df):,} rows"
            )
    except Exception as error:
        st.sidebar.error(f"❌ Could not read the CSV file: {error}")
        df = load_default_sales_data()
        st.sidebar.info("Using the original dataset instead.")
else:
    df = load_default_sales_data()

# ==============================
# 📋 Dataset Information
# ==============================

with st.sidebar:
    st.caption(
        f"Current dataset: {len(df):,} rows × {len(df.columns):,} columns"
    )

    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    with st.expander("🔎 Data Quality Check"):
        st.write(f"Rows: {len(df):,}")
        st.write(f"Columns: {len(df.columns):,}")
        st.write(f"Missing values: {missing_values:,}")
        st.write(f"Duplicate rows: {duplicate_rows:,}")

# ==============================
# 🎛️ Interactive Dashboard Filters
# ==============================

original_df = df.copy()

with st.sidebar:

    st.divider()
    st.markdown("### 🎛️ Dashboard Filters")
    st.caption("Use these filters to update the sales KPIs and sales analysis.")

    # Category filter
    category_options = sorted(original_df["Category"].dropna().unique().tolist()) if "Category" in original_df.columns else []
    selected_categories = st.multiselect(
        "📦 Category",
        category_options,
        default=category_options,
        key="filter_category"
    )

    # Region filter
    region_options = sorted(original_df["Region"].dropna().unique().tolist()) if "Region" in original_df.columns else []
    selected_regions = st.multiselect(
        "🌍 Region",
        region_options,
        default=region_options,
        key="filter_region"
    )

    # Segment filter
    segment_options = sorted(original_df["Segment"].dropna().unique().tolist()) if "Segment" in original_df.columns else []
    selected_segments = st.multiselect(
        "👤 Customer Segment",
        segment_options,
        default=segment_options,
        key="filter_segment"
    )

    # Date filter
    selected_date_range = None
    if "Order Date" in original_df.columns:
        date_series = pd.to_datetime(
            original_df["Order Date"],
            errors="coerce"
        )
        valid_dates = date_series.dropna()

        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()

            selected_date_range = st.date_input(
                "📅 Order Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="filter_date"
            )

# Apply filters
filtered_df = original_df.copy()

if "Category" in filtered_df.columns and selected_categories:
    filtered_df = filtered_df[
        filtered_df["Category"].isin(selected_categories)
    ]

if "Region" in filtered_df.columns and selected_regions:
    filtered_df = filtered_df[
        filtered_df["Region"].isin(selected_regions)
    ]

if "Segment" in filtered_df.columns and selected_segments:
    filtered_df = filtered_df[
        filtered_df["Segment"].isin(selected_segments)
    ]

if (
    "Order Date" in filtered_df.columns
    and selected_date_range
    and isinstance(selected_date_range, tuple)
    and len(selected_date_range) == 2
):
    filtered_df["Order Date"] = pd.to_datetime(
        filtered_df["Order Date"],
        errors="coerce"
    )

    start_date = pd.Timestamp(selected_date_range[0])
    end_date = pd.Timestamp(selected_date_range[1]) + pd.Timedelta(days=1)

    filtered_df = filtered_df[
        (filtered_df["Order Date"] >= start_date)
        & (filtered_df["Order Date"] < end_date)
    ]

# Use filtered data for dashboard sales calculations
if filtered_df.empty:
    st.warning("⚠️ No sales records match the selected filters. Showing the complete dataset.")
    df = original_df.copy()
else:
    df = filtered_df.copy()

# ==============================
# 📅 Current Dataset Season Analysis
# ==============================

season_months = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Monsoon", 10: "Monsoon", 11: "Autumn"
}

dynamic_season_df = df.copy()
dynamic_season_df["Order Date"] = pd.to_datetime(
    dynamic_season_df["Order Date"], errors="coerce"
)
dynamic_season_df["Season"] = dynamic_season_df["Order Date"].dt.month.map(season_months)
valid_season_df = dynamic_season_df.dropna(
    subset=["Season", "Category", "Quantity"]
)

if not valid_season_df.empty:
    season_category_demand = (
        valid_season_df
        .groupby(["Season", "Category"], as_index=False)["Quantity"]
        .sum()
    )
    dynamic_highest_row = season_category_demand.loc[
        season_category_demand["Quantity"].idxmax()
    ]
    dynamic_highest_season = dynamic_highest_row["Season"]
    dynamic_highest_category = dynamic_highest_row["Category"]
    dynamic_highest_quantity = int(dynamic_highest_row["Quantity"])
else:
    dynamic_highest_season = "Unknown"
    dynamic_highest_category = "Unknown"
    dynamic_highest_quantity = 0

# ==============================
# Sidebar
# ==============================

with st.sidebar:

    st.title("🛒 Retail Analytics")

    st.markdown(
        "### Dashboard"
    )

    st.write(
        "Business intelligence dashboard "
        "for retail sales and customer analytics."
    )

    st.divider()

    st.markdown("### 📊 Modules")

    st.write("💰 Sales Performance")
    st.write("👥 Customer Analytics")
    st.write("📈 Sales Forecasting")
    st.write("⏱️ Dwell Time Analytics")
    st.write("📍 Store Zone Analytics")
    st.write("🌦️ Seasonal Demand")
    st.write("💡 Business Insights")
    st.write("📄 Business Report")

    st.divider()

    st.caption(
        "Retail Sales Intelligence System"
    )

    st.caption(
        "AI & ML Based Analytics"
    )
# ==============================
# Sales KPI Dashboard
# ==============================

st.title("🛒 Retail Sales Intelligence Dashboard")

st.markdown(
    """
    **AI & ML Powered Retail Analytics**

    Analyze sales performance, customer behavior,
    demand patterns, and store activity using
    machine learning and computer vision.
    """
)

st.divider()

st.subheader("💰 Sales Performance")

total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_quantity = df["Quantity"].sum()
total_orders = df["Order ID"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Sales",
    f"₹{total_sales:,.2f}"
)

col2.metric(
    "Total Profit",
    f"₹{total_profit:,.2f}"
)

col3.metric(
    "Total Orders",
    f"{total_orders:,}"
)

col4.metric(
    "Total Quantity",
    f"{total_quantity:,}"
)

# ==============================
# 📊 Filtered Sales Analysis
# ==============================

st.subheader("📊 Filtered Sales Analysis")

analysis_col1, analysis_col2, analysis_col3 = st.columns(3)

with analysis_col1:
    if "Category" in df.columns:
        category_sales = (
            df.groupby("Category", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=False)
        )
        fig_category_sales = px.bar(
            category_sales,
            x="Category",
            y="Sales",
            text="Sales",
            title="Sales by Category"
        )
        st.plotly_chart(fig_category_sales, use_container_width=True)

with analysis_col2:
    if "Region" in df.columns:
        region_sales = (
            df.groupby("Region", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=False)
        )
        fig_region_sales = px.bar(
            region_sales,
            x="Region",
            y="Sales",
            text="Sales",
            title="Sales by Region"
        )
        st.plotly_chart(fig_region_sales, use_container_width=True)

with analysis_col3:
    if "Segment" in df.columns:
        segment_sales = (
            df.groupby("Segment", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=False)
        )
        fig_segment_sales = px.bar(
            segment_sales,
            x="Segment",
            y="Sales",
            text="Sales",
            title="Sales by Customer Segment"
        )
        st.plotly_chart(fig_segment_sales, use_container_width=True)

st.caption(
    f"Showing {len(df):,} sales records after applying the selected filters."
)

# ==============================
# 📥 Download Filtered Dataset
# ==============================

st.download_button(
    label="⬇️ Download Filtered Sales Data (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_retail_sales.csv",
    mime="text/csv",
    use_container_width=True
)

# ==============================
# 📊 Data Visualization & Analytics
# ==============================

st.divider()
st.subheader("📊 Data Visualization & Analytics")
st.write(
    "Explore sales trends, profit by category, regional performance, "
    "customer segment performance, and top products."
)

# Sales trend over time
if "Order Date" in df.columns:
    trend_df = df.dropna(subset=["Order Date"]).copy()
    if not trend_df.empty:
        trend_df["Order Date"] = pd.to_datetime(
            trend_df["Order Date"], errors="coerce"
        )
        trend_df = trend_df.dropna(subset=["Order Date"])
        monthly_sales = (
            trend_df.assign(Month_Date=trend_df["Order Date"].dt.to_period("M").dt.to_timestamp())
            .groupby("Month_Date", as_index=False)["Sales"]
            .sum()
        )
        if not monthly_sales.empty:
            fig_sales_trend = px.line(
                monthly_sales,
                x="Month_Date",
                y="Sales",
                markers=True,
                title="Sales Trend Over Time"
            )
            fig_sales_trend.update_layout(
                xaxis_title="Date",
                yaxis_title="Sales"
            )
            st.plotly_chart(fig_sales_trend, use_container_width=True)

# Profit by category
if "Category" in df.columns:
    category_profit = (
        df.groupby("Category", as_index=False)["Profit"]
        .sum()
        .sort_values("Profit", ascending=False)
    )
    if not category_profit.empty:
        fig_profit_category = px.bar(
            category_profit,
            x="Category",
            y="Profit",
            text="Profit",
            title="Profit by Category"
        )
        st.plotly_chart(fig_profit_category, use_container_width=True)

# Region and segment comparison
compare_col1, compare_col2 = st.columns(2)

with compare_col1:
    if "Region" in df.columns:
        region_profit = (
            df.groupby("Region", as_index=False)["Profit"]
            .sum()
            .sort_values("Profit", ascending=False)
        )
        if not region_profit.empty:
            fig_region_profit = px.bar(
                region_profit,
                x="Region",
                y="Profit",
                text="Profit",
                title="Profit by Region"
            )
            st.plotly_chart(fig_region_profit, use_container_width=True)

with compare_col2:
    if "Segment" in df.columns:
        segment_profit = (
            df.groupby("Segment", as_index=False)["Profit"]
            .sum()
            .sort_values("Profit", ascending=False)
        )
        if not segment_profit.empty:
            fig_segment_profit = px.bar(
                segment_profit,
                x="Segment",
                y="Profit",
                text="Profit",
                title="Profit by Customer Segment"
            )
            st.plotly_chart(fig_segment_profit, use_container_width=True)

# Top 10 products, when product information exists
product_column = None
for candidate in ["Product Name", "Product", "Product_Name"]:
    if candidate in df.columns:
        product_column = candidate
        break

if product_column is not None:
    top_products = (
        df.groupby(product_column, as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
        .head(10)
    )
    if not top_products.empty:
        fig_top_products = px.bar(
            top_products.sort_values("Sales"),
            x="Sales",
            y=product_column,
            orientation="h",
            text="Sales",
            title="Top 10 Products by Sales"
        )
        st.plotly_chart(fig_top_products, use_container_width=True)
else:
    st.info(
        "💡 Top 10 Products chart will appear when the uploaded CSV "
        "contains a Product Name or Product column."
    )

# ==============================
# 💹 Profit Margin Analysis
# ==============================

st.divider()
st.subheader("💹 Profit Margin Analysis")
st.write(
    "Analyze overall and group-wise profit margins using the currently filtered dataset."
)

# Profit Margin = (Profit / Sales) × 100
if "Sales" in df.columns and "Profit" in df.columns:
    margin_sales = pd.to_numeric(df["Sales"], errors="coerce")
    margin_profit = pd.to_numeric(df["Profit"], errors="coerce")

    valid_margin_df = pd.DataFrame({
        "Sales": margin_sales,
        "Profit": margin_profit
    }).dropna()

    total_margin_sales = valid_margin_df["Sales"].sum()
    total_margin_profit = valid_margin_df["Profit"].sum()

    if total_margin_sales != 0:
        overall_profit_margin = (
            total_margin_profit / total_margin_sales
        ) * 100
    else:
        overall_profit_margin = 0.0

    margin_col1, margin_col2, margin_col3 = st.columns(3)

    margin_col1.metric(
        "Overall Profit Margin",
        f"{overall_profit_margin:.2f}%"
    )

    margin_col2.metric(
        "Total Sales",
        f"₹{total_margin_sales:,.2f}"
    )

    margin_col3.metric(
        "Total Profit",
        f"₹{total_margin_profit:,.2f}"
    )

    # Category-wise profit margin
    if "Category" in df.columns:
        category_margin = (
            df.groupby("Category", as_index=False)
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
        )
        category_margin["Profit Margin (%)"] = (
            category_margin["Profit"]
            / category_margin["Sales"].replace(0, pd.NA)
            * 100
        )
        category_margin["Profit Margin (%)"] = (
            category_margin["Profit Margin (%)"].fillna(0)
        )

        st.markdown("#### 📦 Profit Margin by Category")

        fig_category_margin = px.bar(
            category_margin.sort_values("Profit Margin (%)", ascending=False),
            x="Category",
            y="Profit Margin (%)",
            text="Profit Margin (%)",
            title="Profit Margin by Category"
        )
        fig_category_margin.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )
        st.plotly_chart(
            fig_category_margin,
            use_container_width=True
        )

    # Region-wise profit margin
    if "Region" in df.columns:
        region_margin = (
            df.groupby("Region", as_index=False)
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
        )
        region_margin["Profit Margin (%)"] = (
            region_margin["Profit"]
            / region_margin["Sales"].replace(0, pd.NA)
            * 100
        )
        region_margin["Profit Margin (%)"] = (
            region_margin["Profit Margin (%)"].fillna(0)
        )

        st.markdown("#### 🌍 Profit Margin by Region")

        fig_region_margin = px.bar(
            region_margin.sort_values("Profit Margin (%)", ascending=False),
            x="Region",
            y="Profit Margin (%)",
            text="Profit Margin (%)",
            title="Profit Margin by Region"
        )
        fig_region_margin.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )
        st.plotly_chart(
            fig_region_margin,
            use_container_width=True
        )

    # Customer-segment-wise profit margin
    if "Segment" in df.columns:
        segment_margin = (
            df.groupby("Segment", as_index=False)
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
        )
        segment_margin["Profit Margin (%)"] = (
            segment_margin["Profit"]
            / segment_margin["Sales"].replace(0, pd.NA)
            * 100
        )
        segment_margin["Profit Margin (%)"] = (
            segment_margin["Profit Margin (%)"].fillna(0)
        )

        st.markdown("#### 👤 Profit Margin by Customer Segment")

        fig_segment_margin = px.bar(
            segment_margin.sort_values("Profit Margin (%)", ascending=False),
            x="Segment",
            y="Profit Margin (%)",
            text="Profit Margin (%)",
            title="Profit Margin by Customer Segment"
        )
        fig_segment_margin.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )
        st.plotly_chart(
            fig_segment_margin,
            use_container_width=True
        )

else:
    st.info("Profit margin analysis needs Sales and Profit columns.")

# ==============================
# 🚨 Sales Anomaly Detection
# ==============================

st.divider()
st.subheader("🚨 Sales Anomaly Detection")
st.write(
    "Automatically identify unusually high or low monthly sales in the current filtered dataset."
)

if "Order Date" in df.columns and "Sales" in df.columns:
    anomaly_df = df[["Order Date", "Sales"]].copy()
    anomaly_df["Order Date"] = pd.to_datetime(anomaly_df["Order Date"], errors="coerce")
    anomaly_df["Sales"] = pd.to_numeric(anomaly_df["Sales"], errors="coerce")
    anomaly_df = anomaly_df.dropna(subset=["Order Date", "Sales"])

    if not anomaly_df.empty:
        monthly_anomaly = (
            anomaly_df.assign(
                Month_Date=anomaly_df["Order Date"].dt.to_period("M").dt.to_timestamp()
            )
            .groupby("Month_Date", as_index=False)["Sales"]
            .sum()
            .sort_values("Month_Date")
        )

        if len(monthly_anomaly) >= 4:
            q1 = monthly_anomaly["Sales"].quantile(0.25)
            q3 = monthly_anomaly["Sales"].quantile(0.75)
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr

            monthly_anomaly["Anomaly"] = monthly_anomaly["Sales"].apply(
                lambda value: (
                    "High Anomaly" if value > upper_limit
                    else "Low Anomaly" if value < lower_limit
                    else "Normal"
                )
            )

            anomalies_only = monthly_anomaly[
                monthly_anomaly["Anomaly"] != "Normal"
            ].copy()

            anomaly_col1, anomaly_col2, anomaly_col3 = st.columns(3)
            anomaly_col1.metric("Months Analyzed", len(monthly_anomaly))
            anomaly_col2.metric("Anomalous Months", len(anomalies_only))
            anomaly_col3.metric(
                "Normal Months",
                len(monthly_anomaly) - len(anomalies_only)
            )

            fig_anomaly = px.line(
                monthly_anomaly,
                x="Month_Date",
                y="Sales",
                markers=True,
                title="Monthly Sales with Anomaly Detection"
            )
            fig_anomaly.update_layout(
                xaxis_title="Month",
                yaxis_title="Sales"
            )
            st.plotly_chart(fig_anomaly, use_container_width=True)

            if not anomalies_only.empty:
                st.warning(
                    f"⚠️ {len(anomalies_only)} unusual sales month(s) detected using the IQR method."
                )
                display_anomalies = anomalies_only.rename(
                    columns={
                        "Month_Date": "Month",
                        "Sales": "Sales",
                        "Anomaly": "Status"
                    }
                )
                display_anomalies["Month"] = display_anomalies["Month"].dt.strftime("%Y-%m")
                st.dataframe(
                    display_anomalies[["Month", "Sales", "Status"]],
                    use_container_width=True
                )
            else:
                st.success(
                    "✅ No unusual monthly sales values were detected in the current filtered dataset."
                )
        else:
            st.info("At least 4 months of valid sales data are needed for anomaly detection.")
    else:
        st.info("Anomaly detection needs valid Order Date and Sales values.")
else:
    st.info("Anomaly detection needs Order Date and Sales columns.")

# ==============================
# 📅 Current Dataset Seasonal Analysis
# ==============================

st.subheader("📅 Sales Demand by Season")

if not valid_season_df.empty:
    season_summary = (
        valid_season_df
        .groupby("Season", as_index=False)["Quantity"]
        .sum()
    )
    fig_dynamic_season = px.bar(
        season_summary,
        x="Season",
        y="Quantity",
        text="Quantity",
        title="Quantity Sold by Season (Current Dataset)"
    )
    st.plotly_chart(fig_dynamic_season, use_container_width=True)
    st.info(
        f"📌 Current dataset: highest quantity sold is "
        f"{dynamic_highest_quantity} units for {dynamic_highest_category} "
        f"during {dynamic_highest_season}."
    )
else:
    st.info("Season analysis needs valid Order Date and Quantity values.")

st.divider()

st.subheader("👥 Computer Vision Customer Analytics")

# Load advanced customer analytics data
customer_df = pd.read_csv(
    "data/processed/customer_entry_exit.csv"
)

# ==============================
# Customer Metrics
# ==============================

total_frames = len(customer_df)

average_occupancy = customer_df[
    "Current_Customers"
].mean()

peak_occupancy = customer_df[
    "Current_Customers"
].max()

minimum_occupancy = customer_df[
    "Current_Customers"
].min()

total_entered = customer_df[
    "Total_Entered"
].max()

total_exited = customer_df[
    "Total_Exited"
].max()

# ==============================
# KPI Cards
# ==============================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Customers Entered",
    total_entered
)

col2.metric(
    "Customers Exited",
    total_exited
)

col3.metric(
    "Peak Occupancy",
    peak_occupancy
)

col4.metric(
    "Average Occupancy",
    f"{average_occupancy:.2f}"
)

# ==============================
# Customer Traffic Summary
# ==============================

customer_flow = total_entered - total_exited

if customer_flow > 0:
    traffic_status = "More customers entered than exited."
elif customer_flow < 0:
    traffic_status = "More customers exited than entered."
else:
    traffic_status = "Customer entry and exit were balanced."

st.info(
    f"👥 **Customer Traffic Summary**\n\n"
    f"{traffic_status} "
    f"Current peak occupancy was **{peak_occupancy} customers**, "
    f"with an average occupancy of **{average_occupancy:.2f}**."
)

# ==============================
# Occupancy Trend
# ==============================

st.subheader("📊 Store Occupancy Trend")

fig_occupancy = px.line(
    customer_df,
    x="Frame",
    y="Current_Customers",
    title="Customer Occupancy Over Time",
    markers=False
)

fig_occupancy.update_layout(
    xaxis_title="Video Frame",
    yaxis_title="Customers"
)

st.plotly_chart(
    fig_occupancy,
    use_container_width=True
)

# ==============================
# Entry / Exit Analytics
# ==============================

st.subheader("🚪 Customer Entry & Exit Analytics")

entry_exit_df = pd.DataFrame({
    "Type": ["Entered", "Exited"],
    "Customers": [
        total_entered,
        total_exited
    ]
})

fig_entry_exit = px.bar(
    entry_exit_df,
    x="Type",
    y="Customers",
    text="Customers",
    title="Customer Entry vs Exit"
)

st.plotly_chart(
    fig_entry_exit,
    use_container_width=True
)
# ==============================
# Sales Forecasting
# ==============================

st.divider()

st.subheader("📈 Sales Forecasting")

# Load trained Random Forest model
rf_model = joblib.load("models/random_forest_model.pkl")

# Get last 12 records from the sales data
forecast_input = df[
    ["Year", "Month", "Quantity", "Discount"]
].tail(12)

# Generate predictions
forecast_prediction = rf_model.predict(forecast_input)

# Create forecast dataframe
forecast_df = pd.DataFrame({
    "Month": range(1, 13),
    "Predicted Sales": forecast_prediction
})

# Display forecast table
st.dataframe(
    forecast_df,
    use_container_width=True
)

# Forecast chart
fig_forecast = px.line(
    forecast_df,
    x="Month",
    y="Predicted Sales",
    markers=True,
    title="Sales Forecast"
)

st.plotly_chart(
    fig_forecast,
    use_container_width=True
)
# ==============================
# Customer Dwell Time Analytics
# ==============================

st.divider()

st.subheader("⏱️ Customer Dwell Time Analytics")

# Load dwell time data
dwell_df = pd.read_csv(
    "data/processed/customer_dwell_time.csv"
)

# ==============================
# Dwell Time Metrics
# ==============================

average_dwell = dwell_df[
    "Dwell_Time_Seconds"
].mean()

maximum_dwell = dwell_df[
    "Dwell_Time_Seconds"
].max()

minimum_dwell = dwell_df[
    "Dwell_Time_Seconds"
].min()

customers_tracked = dwell_df[
    "Customer_ID"
].nunique()

# ==============================
# KPI Cards
# ==============================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Customers Tracked",
    customers_tracked
)

col2.metric(
    "Average Dwell Time",
    f"{average_dwell:.2f} sec"
)

col3.metric(
    "Maximum Dwell Time",
    f"{maximum_dwell:.2f} sec"
)

col4.metric(
    "Minimum Dwell Time",
    f"{minimum_dwell:.2f} sec"
)

# ==============================
# Dwell Time Chart
# ==============================

st.subheader("📊 Customer Dwell Time by Customer")

fig_dwell = px.bar(
    dwell_df,
    x="Customer_ID",
    y="Dwell_Time_Seconds",
    title="Customer Dwell Time",
    text="Dwell_Time_Seconds"
)

fig_dwell.update_layout(
    xaxis_title="Customer ID",
    yaxis_title="Dwell Time (Seconds)"
)

st.plotly_chart(
    fig_dwell,
    use_container_width=True
)

# ==============================
# Dwell Time Details
# ==============================

st.subheader("📋 Customer Dwell Time Details")

st.dataframe(
    dwell_df,
    use_container_width=True
)
# ==============================
# Store Zone Analytics
# ==============================

st.divider()

st.subheader("📍 Store Zone Analytics")

# Load zone analytics data
zone_df = pd.read_csv(
    "data/processed/customer_zone_analytics.csv"
)

# ==============================
# Zone Metrics
# ==============================

average_zone1 = zone_df[
    "Zone_1_Left"
].mean()

average_zone2 = zone_df[
    "Zone_2_Center"
].mean()

average_zone3 = zone_df[
    "Zone_3_Right"
].mean()

peak_zone1 = zone_df[
    "Zone_1_Left"
].max()

peak_zone2 = zone_df[
    "Zone_2_Center"
].max()

peak_zone3 = zone_df[
    "Zone_3_Right"
].max()

# ==============================
# KPI Cards
# ==============================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Zone 1 Avg Customers",
    f"{average_zone1:.2f}"
)

col2.metric(
    "Zone 2 Avg Customers",
    f"{average_zone2:.2f}"
)

col3.metric(
    "Zone 3 Avg Customers",
    f"{average_zone3:.2f}"
)

# ==============================
# Zone Comparison Data
# ==============================

zone_summary = pd.DataFrame({
    "Zone": [
        "Zone 1 - Left",
        "Zone 2 - Center",
        "Zone 3 - Right"
    ],
    "Average Customers": [
        average_zone1,
        average_zone2,
        average_zone3
    ],
    "Peak Customers": [
        peak_zone1,
        peak_zone2,
        peak_zone3
    ]
})

# ==============================
# Zone Comparison Chart
# ==============================

st.subheader("📊 Customer Activity by Store Zone")

fig_zone = px.bar(
    zone_summary,
    x="Zone",
    y="Average Customers",
    text="Average Customers",
    title="Average Customer Activity by Zone"
)

fig_zone.update_layout(
    xaxis_title="Store Zone",
    yaxis_title="Average Customers"
)

st.plotly_chart(
    fig_zone,
    use_container_width=True
)

# ==============================
# Zone Details
# ==============================

st.subheader("📋 Store Zone Details")

st.dataframe(
    zone_summary,
    use_container_width=True
)
# ==============================
# 👥 Customer Behavior & Sales Context
# ==============================

st.divider()
st.subheader("👥 Customer Behavior & Sales Context")
st.write(
    "Connect customer activity metrics with the sales KPIs for the selected dataset. "
    "A true statistical correlation requires a shared date/time or transaction identifier "
    "between customer-tracking data and sales data."
)

behavior_col1, behavior_col2, behavior_col3, behavior_col4 = st.columns(4)

behavior_col1.metric(
    "Filtered Sales",
    f"₹{total_sales:,.2f}"
)
behavior_col2.metric(
    "Filtered Profit",
    f"₹{total_profit:,.2f}"
)
behavior_col3.metric(
    "Average Customer Dwell",
    f"{average_dwell:.2f} sec"
)
behavior_col4.metric(
    "Peak Store Occupancy",
    f"{peak_occupancy:.0f}"
)

# Customer behavior overview table.
behavior_summary = pd.DataFrame({
    "Customer Behavior Metric": [
        "Customers Tracked",
        "Average Dwell Time (sec)",
        "Maximum Dwell Time (sec)",
        "Minimum Dwell Time (sec)",
        "Average Occupancy",
        "Peak Occupancy",
        "Customers Entered",
        "Customers Exited"
    ],
    "Value": [
        customers_tracked,
        round(average_dwell, 2),
        round(maximum_dwell, 2),
        round(minimum_dwell, 2),
        round(average_occupancy, 2),
        int(peak_occupancy),
        int(total_entered),
        int(total_exited)
    ]
})

st.dataframe(
    behavior_summary,
    use_container_width=True,
    hide_index=True
)

if "Sales" in df.columns and len(df) > 1:
    st.info(
        "💡 The dashboard currently presents sales and customer behavior side by side. "
        "Because the customer analytics files do not contain a matching sales date/time "
        "or transaction ID, this version does not claim a statistical sales-customer correlation."
    )

# ==============================
# Seasonal Demand Prediction
# ==============================

st.divider()

st.subheader("🌦️ Seasonal Demand Prediction")

seasonal_prediction_df = pd.read_csv(
    "data/processed/seasonal_demand_predictions.csv"
)

# Display predicted demand
st.subheader("🔮 Predicted Demand by Season")

st.dataframe(
    seasonal_prediction_df,
    use_container_width=True
)

# ==============================
# High-Demand Category
# ==============================

top_seasonal_df = pd.read_csv(
    "data/processed/future_top_category_by_season.csv"
)

st.subheader("⭐ Expected High-Demand Category by Season")

st.dataframe(
    top_seasonal_df,
    use_container_width=True
)

# ==============================
# Seasonal Demand Chart
# ==============================

fig_seasonal_demand = px.bar(
    seasonal_prediction_df,
    x="Season",
    y="Predicted_Quantity",
    color="Category",
    barmode="group",
    title="Predicted Product Demand by Season"
)

fig_seasonal_demand.update_layout(
    xaxis_title="Season",
    yaxis_title="Predicted Quantity"
)

st.plotly_chart(
    fig_seasonal_demand,
    use_container_width=True
)
# ==============================
# Business Insights
# ==============================

st.divider()
# ==============================
# Business Insight Calculations
# ==============================

# Load seasonal demand predictions
seasonal_prediction_df = pd.read_csv(
    "data/processed/seasonal_demand_predictions.csv"
)

# Find highest predicted demand
highest_demand = seasonal_prediction_df.loc[
    seasonal_prediction_df["Predicted_Quantity"].idxmax()
]

highest_category = highest_demand["Category"]
highest_season = highest_demand["Season"]
highest_quantity = int(
    highest_demand["Predicted_Quantity"]
)

# Load store zone analytics
zone_df = pd.read_csv(
    "data/processed/customer_zone_analytics.csv"
)

# Calculate average customers in each zone
zone_averages = {
    "Zone 1 - Left": zone_df["Zone_1_Left"].mean(),
    "Zone 2 - Center": zone_df["Zone_2_Center"].mean(),
    "Zone 3 - Right": zone_df["Zone_3_Right"].mean()
}

# Find busiest zone
busiest_zone = max(
    zone_averages,
    key=zone_averages.get
)

busiest_zone_value = zone_averages[
    busiest_zone
]
# ==============================
# Kannada Season Translation
# ==============================

kannada_seasons = {
    "Winter": "ಚಳಿಗಾಲ",
    "Spring": "ವಸಂತಕಾಲ",
    "Summer": "ಬೇಸಿಗೆ ಕಾಲ",
    "Monsoon": "ಮಳೆಗಾಲ",
    "Autumn": "ಶರತ್ಕಾಲ"
}

kannada_season = kannada_seasons.get(
    highest_season,
    highest_season
)
# ==============================
# Kannada Category Translation
# ==============================

kannada_categories = {
    "Office Supplies": "ಕಚೇರಿ ಬಳಕೆಯ ಸಾಮಗ್ರಿಗಳು",
    "Furniture": "ಪೀಠೋಪಕರಣಗಳು",
    "Technology": "ತಂತ್ರಜ್ಞಾನ ಉತ್ಪನ್ನಗಳು"
}

kannada_category = kannada_categories.get(
    highest_category,
    highest_category
)
# ==============================
# Kannada Zone Translation
# ==============================

kannada_zones = {
    "Zone 1 - Left": "ವಲಯ 1 - ಎಡಭಾಗ",
    "Zone 2 - Center": "ವಲಯ 2 - ಮಧ್ಯಭಾಗ",
    "Zone 3 - Right": "ವಲಯ 3 - ಬಲಭಾಗ"
}
# ==============================
# Hindi Zone Translation
# ==============================

hindi_zones = {
    "Zone 1 - Left": "ज़ोन 1 - बायाँ भाग",
    "Zone 2 - Center": "ज़ोन 2 - मध्य भाग",
    "Zone 3 - Right": "ज़ोन 3 - दायाँ भाग"
}

hindi_zone = hindi_zones.get(
    busiest_zone,
    busiest_zone
)
kannada_zone = kannada_zones.get(
    busiest_zone,
    busiest_zone
)
# ==============================
# Hindi Season Translation
# ==============================

hindi_seasons = {
    "Winter": "सर्दी का मौसम",
    "Spring": "वसंत ऋतु",
    "Summer": "गर्मी का मौसम",
    "Monsoon": "मानसून का मौसम",
    "Autumn": "पतझड़ का मौसम"
}

hindi_season = hindi_seasons.get(
    highest_season,
    highest_season
)

# ==============================
# Hindi Category Translation
# ==============================

hindi_categories = {
    "Office Supplies": "कार्यालय उपयोग की सामग्री",
    "Furniture": "फर्नीचर",
    "Technology": "तकनीकी उत्पाद"
}

hindi_category = hindi_categories.get(
    highest_category,
    highest_category
)
# ==============================
# Language Selection
# ==============================

language = st.selectbox(
    "🌐 Select Language",
    [
        "English",
        "ಕನ್ನಡ (Kannada)",
        "हिन्दी (Hindi)"
    ]
)
# ==============================
# 🎙️ Retail Voice Assistant
# ==============================

st.subheader("🎙️ Retail Voice Assistant")

st.write(
    "Ask natural questions about sales, profit, quantity, orders, "
    "customers, zones, dwell time, demand, forecasts, and business summary."
)

voice_language_codes = {
    "English": "en",
    "ಕನ್ನಡ (Kannada)": "kn",
    "हिन्दी (Hindi)": "hi"
}

voice_language = voice_language_codes.get(language, "en")

spoken_text = speech_to_text(
    language=voice_language,
    start_prompt="🎤 Start Speaking",
    stop_prompt="⏹️ Stop Recording",
    just_once=True,
    use_container_width=True,
    key="retail_voice_input"
)

# ==============================
# Extra Business Calculations
# ==============================

category_profit_df = (
    df.groupby("Category", as_index=False)["Profit"]
    .sum()
    .sort_values("Profit", ascending=False)
)

highest_profit_category = category_profit_df.iloc[0]["Category"]
highest_profit_category_value = category_profit_df.iloc[0]["Profit"]

lowest_zone = min(zone_averages, key=zone_averages.get)
lowest_zone_value = zone_averages[lowest_zone]

current_occupancy = customer_df["Current_Customers"].iloc[-1]

average_sales = df["Sales"].mean()

# ==============================
# Profit Margin & Comparison Calculations
# ==============================

if total_sales != 0:
    overall_profit_margin = (total_profit / total_sales) * 100
else:
    overall_profit_margin = 0.0


def build_comparison(group_column, value_a, value_b):
    """Return sales, profit, quantity and profit margin for two selected groups."""
    comparison = (
        df[df[group_column].isin([value_a, value_b])]
        .groupby(group_column, as_index=False)
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum")
        )
    )

    comparison["Profit Margin (%)"] = (
        comparison["Profit"]
        / comparison["Sales"].replace(0, pd.NA)
        * 100
    )
    comparison["Profit Margin (%)"] = comparison["Profit Margin (%)"].fillna(0)
    return comparison


# ==============================
# Intent Examples
# ==============================

intent_examples = {
    "TOTAL_SALES": [
        "What is the total sales?", "How much sales did we make?",
        "Tell me the total amount of sales", "ಒಟ್ಟು ಮಾರಾಟ ಎಷ್ಟು?",
        "कुल बिक्री कितनी है?"
    ],
    "TOTAL_PROFIT": [
        "What is the total profit?", "How much profit did we make?",
        "Tell me the profit", "ಒಟ್ಟು ಲಾಭ ಎಷ್ಟು?", "कुल लाभ कितना है?"
    ],
    "TOTAL_QUANTITY": [
        "How many products were sold?", "What is the total quantity sold?",
        "Tell me total quantity", "ಒಟ್ಟು ಎಷ್ಟು ಉತ್ಪನ್ನಗಳು ಮಾರಾಟವಾಗಿವೆ?",
        "कुल कितने उत्पाद बिके?"
    ],
    "TOTAL_ORDERS": [
        "How many orders are there?", "What is the total number of orders?",
        "Tell me total orders", "ಒಟ್ಟು ಆರ್ಡರ್‌ಗಳು ಎಷ್ಟು?", "कुल ऑर्डर कितने हैं?"
    ],
    "AVERAGE_SALES": [
        "What is the average sales?", "What is the average sale amount?",
        "Tell me average sales", "ಸರಾಸರಿ ಮಾರಾಟ ಎಷ್ಟು?", "औसत बिक्री कितनी है?"
    ],
    "PROFIT_MARGIN": [
        "What is the profit margin?", "What is our overall profit margin?",
        "Tell me the profit margin", "How much profit margin do we have?",
        "What percentage of sales is profit?", "ಲಾಭದ ಅಂಚು ಎಷ್ಟು?",
        "ಒಟ್ಟು ಲಾಭದ ಅಂಚು ಎಷ್ಟು?", "लाभ मार्जिन कितना है?",
        "कुल लाभ मार्जिन कितना है?"
    ],

    "CATEGORY_COMPARISON": [
        "Compare Technology and Furniture",
        "Compare two categories", "Compare categories",
        "Which is better between Technology and Furniture?",
        "Which category has more sales?",
        "Compare product categories", "ವರ್ಗಗಳನ್ನು ಹೋಲಿಸಿ",
        "ತಂತ್ರಜ್ಞಾನ ಮತ್ತು ಪೀಠೋಪಕರಣಗಳನ್ನು ಹೋಲಿಸಿ",
        "टेक्नोलॉजी और फर्नीचर की तुलना करो",
        "दो श्रेणियों की तुलना करो"
    ],

    "REGION_COMPARISON": [
        "Compare West and East", "Compare two regions",
        "Compare regions", "Which region has more sales?",
        "Compare regional performance", "ಪ್ರದೇಶಗಳನ್ನು ಹೋಲಿಸಿ",
        "ವೆಸ್ಟ್ ಮತ್ತು ಈಸ್ಟ್ ಹೋಲಿಸಿ", "वेस्ट और ईस्ट की तुलना करो",
        "दो क्षेत्रों की तुलना करो"
    ],

    "SEGMENT_COMPARISON": [
        "Compare Consumer and Corporate",
        "Compare two customer segments",
        "Compare customer segments", "Which segment has more sales?",
        "Compare segment performance", "ಗ್ರಾಹಕರ ವಿಭಾಗಗಳನ್ನು ಹೋಲಿಸಿ",
        "ಕನ್ಸ್ಯೂಮರ್ ಮತ್ತು ಕಾರ್ಪೊರೇಟ್ ಹೋಲಿಸಿ",
        "कंज्यूमर और कॉर्पोरेट की तुलना करो",
        "दो ग्राहक सेगमेंट की तुलना करो"
    ],
    "HIGHEST_DEMAND": [
        "Which product has the highest demand?", "What is the most demanded product?",
        "Which product sells the most?", "ಹೆಚ್ಚಿನ ಬೇಡಿಕೆ ಇರುವ ಉತ್ಪನ್ನ ಯಾವುದು?",
        "सबसे ज्यादा मांग किस उत्पाद की है?"
    ],
    "HIGHEST_PROFIT_CATEGORY": [
        "Which category has the highest profit?", "Which product category makes the most profit?",
        "What category generated the most profit?", "ಹೆಚ್ಚು ಲಾಭ ಕೊಡುವ ವರ್ಗ ಯಾವುದು?",
        "सबसे ज्यादा लाभ किस श्रेणी से है?"
    ],
    "CUSTOMERS_ENTERED": [
        "How many customers entered?", "How many customers came to the store?",
        "Tell me how many customers entered", "ಎಷ್ಟು ಗ್ರಾಹಕರು ಅಂಗಡಿಗೆ ಬಂದಿದ್ದಾರೆ?",
        "कितने ग्राहक दुकान में आए?"
    ],
    "CUSTOMERS_EXITED": [
        "How many customers exited?", "How many customers left the store?",
        "Tell me the number of customers who exited", "ಎಷ್ಟು ಗ್ರಾಹಕರು ಹೊರಗೆ ಹೋಗಿದ್ದಾರೆ?",
        "कितने ग्राहक बाहर गए?"
    ],
    "CURRENT_OCCUPANCY": [
        "How many customers are currently in the store?", "How many people are in the store now?",
        "What is the current occupancy?", "ಈಗ ಅಂಗಡಿಯಲ್ಲಿ ಎಷ್ಟು ಗ್ರಾಹಕರಿದ್ದಾರೆ?",
        "अभी स्टोर में कितने ग्राहक हैं?"
    ],
    "PEAK_OCCUPANCY": [
        "What is the peak occupancy?", "What was the maximum number of customers in the store?",
        "Tell me the highest occupancy", "ಗರಿಷ್ಠ ಗ್ರಾಹಕರ ಸಂಖ್ಯೆ ಎಷ್ಟು?",
        "अधिकतम ग्राहक संख्या कितनी थी?"
    ],
    "AVERAGE_OCCUPANCY": [
        "What is the average occupancy?", "Tell me the average customer occupancy",
        "What is the average number of customers?", "ಸರಾಸರಿ ಗ್ರಾಹಕರ ಸಂಖ್ಯೆ ಎಷ್ಟು?",
        "औसत ग्राहक संख्या कितनी है?"
    ],
    "BUSIEST_ZONE": [
        "Which zone is busiest?", "Where is the most customer activity?",
        "Which area has the highest customer activity?", "ಯಾವ ವಲಯದಲ್ಲಿ ಹೆಚ್ಚು ಗ್ರಾಹಕರ ಚಟುವಟಿಕೆ ಇದೆ?",
        "सबसे व्यस्त क्षेत्र कौन सा है?"
    ],
    "LOWEST_ACTIVITY_ZONE": [
        "Which zone has the lowest activity?", "Which area has the fewest customers?",
        "Where is customer activity lowest?", "ಕಡಿಮೆ ಗ್ರಾಹಕರ ಚಟುವಟಿಕೆ ಇರುವ ವಲಯ ಯಾವುದು?",
        "सबसे कम ग्राहक गतिविधि वाला क्षेत्र कौन सा है?"
    ],
    "AVERAGE_DWELL": [
        "What is the average dwell time?", "How long do customers stay on average?",
        "Tell me average customer dwell time", "ಸರಾಸರಿ ಡ್ವೆಲ್ ಸಮಯ ಎಷ್ಟು?",
        "औसत ड्वेल टाइम कितना है?"
    ],
    "MAXIMUM_DWELL": [
        "What is the maximum dwell time?", "What is the longest time a customer stayed?",
        "Tell me the maximum dwell time", "ಗರಿಷ್ಠ ಡ್ವೆಲ್ ಸಮಯ ಎಷ್ಟು?",
        "अधिकतम ड्वेल टाइम कितना है?"
    ],
    "HIGHEST_DEMAND_SEASON": [
        "Which season has the highest demand?", "What season has the most demand?",
        "Which season sold the most quantity?", "What season has the most sales?",
        "Tell me the season with highest demand", "ಹೆಚ್ಚಿನ ಬೇಡಿಕೆ ಇರುವ ಋತು ಯಾವುದು?",
        "सबसे ज्यादा मांग किस मौसम में है?", "किस मौसम में सबसे ज्यादा बिक्री है?"
    ],
    "SALES_FORECAST": [
        "What is the sales forecast?", "Tell me the predicted sales",
        "What are the forecasted sales?", "ಮಾರಾಟದ ಮುನ್ಸೂಚನೆ ಎಷ್ಟು?",
        "बिक्री का पूर्वानुमान क्या है?"
    ],
    "BUSINESS_SUMMARY": [
        "Give me a complete business summary", "Give me the business summary",
        "Summarize the retail dashboard", "ವ್ಯವಹಾರದ ಸಂಪೂರ್ಣ ಸಾರಾಂಶ ನೀಡಿ",
        "मुझे पूरा बिजनेस सारांश बताओ"
    ]
}

@st.cache_resource
def build_intent_embeddings():
    return {
        intent_name: intent_model.encode(examples)
        for intent_name, examples in intent_examples.items()
    }

intent_example_embeddings = build_intent_embeddings()


def understand_intent(question):
    question_embedding = intent_model.encode(question)
    best_intent = "UNKNOWN"
    best_score = -1

    for intent_name, embeddings in intent_example_embeddings.items():
        for embedding in embeddings:
            denominator = (
                (question_embedding @ question_embedding) ** 0.5
                * (embedding @ embedding) ** 0.5
            )
            if denominator == 0:
                continue
            score = (question_embedding @ embedding) / denominator
            if score > best_score:
                best_score = score
                best_intent = intent_name

    return best_intent, best_score


if spoken_text:

    st.success(f"🎤 You said: {spoken_text}")
    intent, confidence = understand_intent(spoken_text)

    st.info(
        f"🧠 Detected Intent: {intent} | Confidence: {confidence:.2f}"
    )

    if confidence < 0.45:
        if language == "English":
            answer_text = (
                "Sorry, I could not clearly understand the question. "
                "Please ask about sales, profit, quantity, orders, customers, "
                "zones, dwell time, demand, forecast, or business summary."
            )
        elif language == "ಕನ್ನಡ (Kannada)":
            answer_text = (
                "ಕ್ಷಮಿಸಿ, ಪ್ರಶ್ನೆಯನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. "
                "ಮಾರಾಟ, ಲಾಭ, ಪ್ರಮಾಣ, ಆರ್ಡರ್‌ಗಳು, ಗ್ರಾಹಕರು, ವಲಯ, ಡ್ವೆಲ್ ಸಮಯ, "
                "ಬೇಡಿಕೆ, ಮುನ್ಸೂಚನೆ ಅಥವಾ ವ್ಯವಹಾರದ ಸಾರಾಂಶದ ಬಗ್ಗೆ ಕೇಳಿ."
            )
        else:
            answer_text = (
                "क्षमा करें, मैं प्रश्न को स्पष्ट रूप से समझ नहीं पाया। "
                "बिक्री, लाभ, मात्रा, ऑर्डर, ग्राहक, ज़ोन, ड्वेल टाइम, "
                "मांग, पूर्वानुमान या बिजनेस सारांश के बारे में पूछें।"
            )
        st.warning(f"🤖 {answer_text}")
        speak_text(answer_text, language)

    else:
        if intent == "PROFIT_MARGIN":
            if language == "English":
                answer_text = (
                    f"The overall profit margin is {overall_profit_margin:.2f}%. "
                    f"Profit margin is calculated as profit divided by sales multiplied by 100."
                )
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = (
                    f"ಒಟ್ಟು ಲಾಭದ ಅಂಚು {overall_profit_margin:.2f}% ಆಗಿದೆ. "
                    f"ಲಾಭದ ಅಂಚು = ಲಾಭವನ್ನು ಮಾರಾಟದಿಂದ ಭಾಗಿಸಿ 100ರಿಂದ ಗುಣಿಸಿದ ಮೌಲ್ಯ."
                )
            else:
                answer_text = (
                    f"कुल लाभ मार्जिन {overall_profit_margin:.2f}% है। "
                    f"लाभ मार्जिन = लाभ को बिक्री से भाग देकर 100 से गुणा किया जाता है।"
                )

        elif intent == "CATEGORY_COMPARISON":
            categories = df["Category"].dropna().astype(str).unique().tolist()
            if len(categories) >= 2:
                # Prefer Technology and Furniture when both exist; otherwise use the first two categories.
                preferred = [c for c in ["Technology", "Furniture"] if c in categories]
                if len(preferred) == 2:
                    category_a, category_b = preferred
                else:
                    category_a, category_b = categories[:2]

                comparison = build_comparison("Category", category_a, category_b)

                def row_for(name):
                    row = comparison[comparison["Category"] == name]
                    return row.iloc[0] if not row.empty else None

                a = row_for(category_a)
                b = row_for(category_b)

                if a is not None and b is not None:
                    if language == "English":
                        answer_text = (
                            f"Comparison of {category_a} and {category_b}: "
                            f"{category_a} sales ₹{a['Sales']:,.2f}, profit ₹{a['Profit']:,.2f}, "
                            f"quantity {int(a['Quantity']):,}, margin {a['Profit Margin (%)']:.2f}%. "
                            f"{category_b} sales ₹{b['Sales']:,.2f}, profit ₹{b['Profit']:,.2f}, "
                            f"quantity {int(b['Quantity']):,}, margin {b['Profit Margin (%)']:.2f}%."
                        )
                    elif language == "ಕನ್ನಡ (Kannada)":
                        answer_text = (
                            f"{category_a} ಮತ್ತು {category_b} ಹೋಲಿಕೆ: "
                            f"{category_a} ಮಾರಾಟ ₹{a['Sales']:,.2f}, ಲಾಭ ₹{a['Profit']:,.2f}, "
                            f"ಪ್ರಮಾಣ {int(a['Quantity']):,}, ಲಾಭದ ಅಂಚು {a['Profit Margin (%)']:.2f}%. "
                            f"{category_b} ಮಾರಾಟ ₹{b['Sales']:,.2f}, ಲಾಭ ₹{b['Profit']:,.2f}, "
                            f"ಪ್ರಮಾಣ {int(b['Quantity']):,}, ಲಾಭದ ಅಂಚು {b['Profit Margin (%)']:.2f}%."
                        )
                    else:
                        answer_text = (
                            f"{category_a} और {category_b} की तुलना: "
                            f"{category_a} बिक्री ₹{a['Sales']:,.2f}, लाभ ₹{a['Profit']:,.2f}, "
                            f"मात्रा {int(a['Quantity']):,}, लाभ मार्जिन {a['Profit Margin (%)']:.2f}%. "
                            f"{category_b} बिक्री ₹{b['Sales']:,.2f}, लाभ ₹{b['Profit']:,.2f}, "
                            f"मात्रा {int(b['Quantity']):,}, लाभ मार्जिन {b['Profit Margin (%)']:.2f}%."
                        )
                else:
                    answer_text = "I could not find enough data to compare the two categories."
            else:
                answer_text = "At least two categories are needed for comparison."

        elif intent == "REGION_COMPARISON":
            regions = df["Region"].dropna().astype(str).unique().tolist()
            if len(regions) >= 2:
                preferred = [r for r in ["West", "East"] if r in regions]
                if len(preferred) == 2:
                    region_a, region_b = preferred
                else:
                    region_a, region_b = regions[:2]

                comparison = build_comparison("Region", region_a, region_b)
                a = comparison[comparison["Region"] == region_a].iloc[0]
                b = comparison[comparison["Region"] == region_b].iloc[0]

                if language == "English":
                    answer_text = (
                        f"Comparison of {region_a} and {region_b}: "
                        f"{region_a} sales ₹{a['Sales']:,.2f}, profit ₹{a['Profit']:,.2f}, "
                        f"quantity {int(a['Quantity']):,}, margin {a['Profit Margin (%)']:.2f}%. "
                        f"{region_b} sales ₹{b['Sales']:,.2f}, profit ₹{b['Profit']:,.2f}, "
                        f"quantity {int(b['Quantity']):,}, margin {b['Profit Margin (%)']:.2f}%."
                    )
                elif language == "ಕನ್ನಡ (Kannada)":
                    answer_text = (
                        f"{region_a} ಮತ್ತು {region_b} ಹೋಲಿಕೆ: "
                        f"{region_a} ಮಾರಾಟ ₹{a['Sales']:,.2f}, ಲಾಭ ₹{a['Profit']:,.2f}, "
                        f"ಪ್ರಮಾಣ {int(a['Quantity']):,}, ಲಾಭದ ಅಂಚು {a['Profit Margin (%)']:.2f}%. "
                        f"{region_b} ಮಾರಾಟ ₹{b['Sales']:,.2f}, ಲಾಭ ₹{b['Profit']:,.2f}, "
                        f"ಪ್ರಮಾಣ {int(b['Quantity']):,}, ಲಾಭದ ಅಂಚು {b['Profit Margin (%)']:.2f}%."
                    )
                else:
                    answer_text = (
                        f"{region_a} और {region_b} की तुलना: "
                        f"{region_a} बिक्री ₹{a['Sales']:,.2f}, लाभ ₹{a['Profit']:,.2f}, "
                        f"मात्रा {int(a['Quantity']):,}, लाभ मार्जिन {a['Profit Margin (%)']:.2f}%. "
                        f"{region_b} बिक्री ₹{b['Sales']:,.2f}, लाभ ₹{b['Profit']:,.2f}, "
                        f"मात्रा {int(b['Quantity']):,}, लाभ मार्जिन {b['Profit Margin (%)']:.2f}%."
                    )
            else:
                answer_text = "At least two regions are needed for comparison."

        elif intent == "SEGMENT_COMPARISON":
            segments = df["Segment"].dropna().astype(str).unique().tolist()
            if len(segments) >= 2:
                preferred = [s for s in ["Consumer", "Corporate"] if s in segments]
                if len(preferred) == 2:
                    segment_a, segment_b = preferred
                else:
                    segment_a, segment_b = segments[:2]

                comparison = build_comparison("Segment", segment_a, segment_b)
                a = comparison[comparison["Segment"] == segment_a].iloc[0]
                b = comparison[comparison["Segment"] == segment_b].iloc[0]

                if language == "English":
                    answer_text = (
                        f"Comparison of {segment_a} and {segment_b}: "
                        f"{segment_a} sales ₹{a['Sales']:,.2f}, profit ₹{a['Profit']:,.2f}, "
                        f"quantity {int(a['Quantity']):,}, margin {a['Profit Margin (%)']:.2f}%. "
                        f"{segment_b} sales ₹{b['Sales']:,.2f}, profit ₹{b['Profit']:,.2f}, "
                        f"quantity {int(b['Quantity']):,}, margin {b['Profit Margin (%)']:.2f}%."
                    )
                elif language == "ಕನ್ನಡ (Kannada)":
                    answer_text = (
                        f"{segment_a} ಮತ್ತು {segment_b} ಹೋಲಿಕೆ: "
                        f"{segment_a} ಮಾರಾಟ ₹{a['Sales']:,.2f}, ಲಾಭ ₹{a['Profit']:,.2f}, "
                        f"ಪ್ರಮಾಣ {int(a['Quantity']):,}, ಲಾಭದ ಅಂಚು {a['Profit Margin (%)']:.2f}%. "
                        f"{segment_b} ಮಾರಾಟ ₹{b['Sales']:,.2f}, ಲಾಭ ₹{b['Profit']:,.2f}, "
                        f"ಪ್ರಮಾಣ {int(b['Quantity']):,}, ಲಾಭದ ಅಂಚು {b['Profit Margin (%)']:.2f}%."
                    )
                else:
                    answer_text = (
                        f"{segment_a} और {segment_b} की तुलना: "
                        f"{segment_a} बिक्री ₹{a['Sales']:,.2f}, लाभ ₹{a['Profit']:,.2f}, "
                        f"मात्रा {int(a['Quantity']):,}, लाभ मार्जिन {a['Profit Margin (%)']:.2f}%. "
                        f"{segment_b} बिक्री ₹{b['Sales']:,.2f}, लाभ ₹{b['Profit']:,.2f}, "
                        f"मात्रा {int(b['Quantity']):,}, लाभ मार्जिन {b['Profit Margin (%)']:.2f}%."
                    )
            else:
                answer_text = "At least two customer segments are needed for comparison."

        elif intent == "TOTAL_SALES":
            if language == "English":
                answer_text = f"The total sales are ₹{total_sales:,.2f}."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"ಒಟ್ಟು ಮಾರಾಟವು ₹{total_sales:,.2f} ರೂಪಾಯಿಗಳು."
            else:
                answer_text = f"कुल बिक्री ₹{total_sales:,.2f} रुपये है।"

        elif intent == "TOTAL_PROFIT":
            if language == "English":
                answer_text = f"The total profit is ₹{total_profit:,.2f}."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"ಒಟ್ಟು ಲಾಭವು ₹{total_profit:,.2f} ರೂಪಾಯಿಗಳು."
            else:
                answer_text = f"कुल लाभ ₹{total_profit:,.2f} रुपये है।"

        elif intent == "TOTAL_QUANTITY":
            if language == "English":
                answer_text = f"The total quantity sold is {total_quantity:,} units."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"ಒಟ್ಟು ಮಾರಾಟವಾದ ಪ್ರಮಾಣ {total_quantity:,} ಘಟಕಗಳು."
            else:
                answer_text = f"कुल बिक्री की मात्रा {total_quantity:,} यूनिट है।"

        elif intent == "TOTAL_ORDERS":
            if language == "English":
                answer_text = f"The total number of orders is {total_orders:,}."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"ಒಟ್ಟು ಆರ್ಡರ್‌ಗಳ ಸಂಖ್ಯೆ {total_orders:,}."
            else:
                answer_text = f"कुल ऑर्डर की संख्या {total_orders:,} है।"

        elif intent == "AVERAGE_SALES":
            if language == "English":
                answer_text = f"The average sales value per record is ₹{average_sales:,.2f}."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"ಪ್ರತಿ ದಾಖಲೆಯ ಸರಾಸರಿ ಮಾರಾಟ ಮೌಲ್ಯ ₹{average_sales:,.2f}."
            else:
                answer_text = f"प्रति रिकॉर्ड औसत बिक्री मूल्य ₹{average_sales:,.2f} है।"

        elif intent == "HIGHEST_DEMAND":
            if language == "English":
                answer_text = f"The highest predicted demand is for {highest_category} during {highest_season}. The expected quantity is {highest_quantity} units."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"{kannada_season} ಸಮಯದಲ್ಲಿ {kannada_category}ಗಳಿಗೆ ಅತ್ಯಧಿಕ ನಿರೀಕ್ಷಿತ ಬೇಡಿಕೆ ಇದೆ. ನಿರೀಕ್ಷಿತ ಪ್ರಮಾಣ {highest_quantity} ಘಟಕಗಳು."
            else:
                answer_text = f"{hindi_season} के दौरान {hindi_category} की सबसे अधिक अनुमानित मांग है। अनुमानित मात्रा {highest_quantity} यूनिट है।"

        elif intent == "HIGHEST_PROFIT_CATEGORY":
            if language == "English":
                answer_text = f"The category with the highest total profit is {highest_profit_category}, with ₹{highest_profit_category_value:,.2f} profit."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"ಅತ್ಯಧಿಕ ಒಟ್ಟು ಲಾಭ ನೀಡಿದ ವರ್ಗ {highest_profit_category}. ಲಾಭ ₹{highest_profit_category_value:,.2f}."
            else:
                answer_text = f"सबसे अधिक कुल लाभ वाली श्रेणी {highest_profit_category} है, जिसका लाभ ₹{highest_profit_category_value:,.2f} है।"

        elif intent == "CUSTOMERS_ENTERED":
            if language == "English":
                answer_text = f"{total_entered} customers entered the store."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"ಅಂಗಡಿಗೆ {total_entered} ಗ್ರಾಹಕರು ಒಳಗೆ ಬಂದಿದ್ದಾರೆ."
            else:
                answer_text = f"स्टोर में {total_entered} ग्राहक अंदर आए हैं।"

        elif intent == "CUSTOMERS_EXITED":
            if language == "English":
                answer_text = f"{total_exited} customers exited the store."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"ಅಂಗಡಿಯಿಂದ {total_exited} ಗ್ರಾಹಕರು ಹೊರಗೆ ಹೋಗಿದ್ದಾರೆ."
            else:
                answer_text = f"स्टोर से {total_exited} ग्राहक बाहर गए हैं।"

        elif intent == "CURRENT_OCCUPANCY":
            if language == "English":
                answer_text = f"The latest recorded store occupancy is {current_occupancy} customers."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"ಇತ್ತೀಚಿನ ದಾಖಲೆಯ ಪ್ರಕಾರ ಅಂಗಡಿಯಲ್ಲಿ {current_occupancy} ಗ್ರಾಹಕರಿದ್ದಾರೆ."
            else:
                answer_text = f"नवीनतम रिकॉर्ड के अनुसार स्टोर में {current_occupancy} ग्राहक हैं।"

        elif intent == "PEAK_OCCUPANCY":
            if language == "English":
                answer_text = f"The peak store occupancy was {peak_occupancy} customers."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"ಅಂಗಡಿಯ ಗರಿಷ್ಠ ಆಕ್ಯುಪೆನ್ಸಿ {peak_occupancy} ಗ್ರಾಹಕರು."
            else:
                answer_text = f"स्टोर की अधिकतम ऑक्यूपेंसी {peak_occupancy} ग्राहक थी।"

        elif intent == "AVERAGE_OCCUPANCY":
            if language == "English":
                answer_text = f"The average customer occupancy is {average_occupancy:.2f} customers."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"ಸರಾಸರಿ ಗ್ರಾಹಕರ ಸಂಖ್ಯೆ {average_occupancy:.2f} ಆಗಿದೆ."
            else:
                answer_text = f"औसत ग्राहक संख्या {average_occupancy:.2f} है।"

        elif intent == "BUSIEST_ZONE":
            if language == "English":
                answer_text = f"{busiest_zone} has the highest customer activity. The average activity is {busiest_zone_value:.2f} customers."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"{kannada_zone} ನಲ್ಲಿ ಹೆಚ್ಚಿನ ಗ್ರಾಹಕರ ಚಟುವಟಿಕೆ ಇದೆ. ಸರಾಸರಿ ಗ್ರಾಹಕರ ಸಂಖ್ಯೆ {busiest_zone_value:.2f}."
            else:
                answer_text = f"{hindi_zone} में सबसे अधिक ग्राहक गतिविधि है। औसत ग्राहक गतिविधि {busiest_zone_value:.2f} है।"

        elif intent == "LOWEST_ACTIVITY_ZONE":
            lowest_zone_kn = kannada_zones.get(lowest_zone, lowest_zone)
            lowest_zone_hi = hindi_zones.get(lowest_zone, lowest_zone)
            if language == "English":
                answer_text = f"{lowest_zone} has the lowest customer activity, with an average of {lowest_zone_value:.2f} customers."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"{lowest_zone_kn} ನಲ್ಲಿ ಕಡಿಮೆ ಗ್ರಾಹಕರ ಚಟುವಟಿಕೆ ಇದೆ. ಸರಾಸರಿ {lowest_zone_value:.2f} ಗ್ರಾಹಕರು."
            else:
                answer_text = f"{lowest_zone_hi} में सबसे कम ग्राहक गतिविधि है। औसत {lowest_zone_value:.2f} ग्राहक हैं।"

        elif intent == "AVERAGE_DWELL":
            if language == "English":
                answer_text = f"The average customer dwell time is {average_dwell:.2f} seconds."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"ಸರಾಸರಿ ಗ್ರಾಹಕರ ಡ್ವೆಲ್ ಸಮಯ {average_dwell:.2f} ಸೆಕೆಂಡುಗಳು."
            else:
                answer_text = f"औसत ग्राहक ड्वेल टाइम {average_dwell:.2f} सेकंड है।"

        elif intent == "MAXIMUM_DWELL":
            if language == "English":
                answer_text = f"The maximum customer dwell time is {maximum_dwell:.2f} seconds."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"ಗರಿಷ್ಠ ಗ್ರಾಹಕರ ಡ್ವೆಲ್ ಸಮಯ {maximum_dwell:.2f} ಸೆಕೆಂಡುಗಳು."
            else:
                answer_text = f"अधिकतम ग्राहक ड्वेल टाइम {maximum_dwell:.2f} सेकंड है।"

        elif intent == "HIGHEST_DEMAND_SEASON":
            dynamic_kn_season = kannada_seasons.get(
                dynamic_highest_season, dynamic_highest_season
            )
            dynamic_hi_season = hindi_seasons.get(
                dynamic_highest_season, dynamic_highest_season
            )
            dynamic_kn_category = kannada_categories.get(
                dynamic_highest_category, dynamic_highest_category
            )
            dynamic_hi_category = hindi_categories.get(
                dynamic_highest_category, dynamic_highest_category
            )

            if language == "English":
                answer_text = (
                    f"Based on the current dataset, the highest quantity sold "
                    f"is in {dynamic_highest_season}, for {dynamic_highest_category}, "
                    f"with {dynamic_highest_quantity} units."
                )
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = (
                    f"ಪ್ರಸ್ತುತ ಡೇಟಾಸೆಟ್ ಪ್ರಕಾರ, ಅತ್ಯಧಿಕ ಪ್ರಮಾಣದ ಮಾರಾಟ "
                    f"{dynamic_kn_season} ಸಮಯದಲ್ಲಿ {dynamic_kn_category}ಗೆ ಇದೆ. "
                    f"ಮಾರಾಟವಾದ ಪ್ರಮಾಣ {dynamic_highest_quantity} ಘಟಕಗಳು."
                )
            else:
                answer_text = (
                    f"वर्तमान डेटासेट के अनुसार, सबसे अधिक मात्रा में बिक्री "
                    f"{dynamic_hi_season} में {dynamic_hi_category} की हुई है। "
                    f"बिक्री की मात्रा {dynamic_highest_quantity} यूनिट है।"
                )

        elif intent == "SALES_FORECAST":
            forecast_values = forecast_df["Predicted Sales"].tolist()
            first_forecast = forecast_values[0]
            average_forecast = sum(forecast_values) / len(forecast_values)
            if language == "English":
                answer_text = f"The sales forecast contains 12 predicted values. The first predicted sales value is ₹{first_forecast:,.2f}, and the average predicted sales is ₹{average_forecast:,.2f}."
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = f"ಮಾರಾಟದ ಮುನ್ಸೂಚನೆಯಲ್ಲಿ 12 ನಿರೀಕ್ಷಿತ ಮೌಲ್ಯಗಳಿವೆ. ಮೊದಲ ನಿರೀಕ್ಷಿತ ಮಾರಾಟ ಮೌಲ್ಯ ₹{first_forecast:,.2f}, ಮತ್ತು ಸರಾಸರಿ ನಿರೀಕ್ಷಿತ ಮಾರಾಟ ₹{average_forecast:,.2f}."
            else:
                answer_text = f"बिक्री पूर्वानुमान में 12 अनुमानित मान हैं। पहला अनुमानित बिक्री मान ₹{first_forecast:,.2f} है, और औसत अनुमानित बिक्री ₹{average_forecast:,.2f} है।"

        elif intent == "BUSINESS_SUMMARY":
            if language == "English":
                answer_text = (
                    f"Business summary: total sales are ₹{total_sales:,.2f}, total profit is ₹{total_profit:,.2f}, "
                    f"total orders are {total_orders:,}, and total quantity sold is {total_quantity:,} units. "
                    f"The busiest zone is {busiest_zone}, average occupancy is {average_occupancy:.2f}, "
                    f"and the highest current demand is for {dynamic_highest_category} during {dynamic_highest_season}."
                )
            elif language == "ಕನ್ನಡ (Kannada)":
                answer_text = (
                    f"ವ್ಯವಹಾರದ ಸಾರಾಂಶ: ಒಟ್ಟು ಮಾರಾಟ ₹{total_sales:,.2f}, ಒಟ್ಟು ಲಾಭ ₹{total_profit:,.2f}, "
                    f"ಒಟ್ಟು ಆರ್ಡರ್‌ಗಳು {total_orders:,}, ಮತ್ತು ಮಾರಾಟವಾದ ಒಟ್ಟು ಪ್ರಮಾಣ {total_quantity:,} ಘಟಕಗಳು. "
                    f"ಹೆಚ್ಚಿನ ಗ್ರಾಹಕರ ಚಟುವಟಿಕೆ {kannada_zone} ನಲ್ಲಿ ಇದೆ. ಸರಾಸರಿ ಗ್ರಾಹಕರ ಸಂಖ್ಯೆ {average_occupancy:.2f}. "
                    f"ಅತ್ಯಧಿಕ ನಿರೀಕ್ಷಿತ ಬೇಡಿಕೆ {kannada_category}ಗಳಿಗೆ {kannada_season} ಸಮಯದಲ್ಲಿ ಇದೆ."
                )
            else:
                answer_text = (
                    f"बिजनेस सारांश: कुल बिक्री ₹{total_sales:,.2f}, कुल लाभ ₹{total_profit:,.2f}, "
                    f"कुल ऑर्डर {total_orders:,}, और कुल बिक्री मात्रा {total_quantity:,} यूनिट है। "
                    f"सबसे अधिक ग्राहक गतिविधि {hindi_zone} में है। औसत ग्राहक संख्या {average_occupancy:.2f} है। "
                    f"सबसे अधिक अनुमानित मांग {hindi_category} की {hindi_season} में है।"
                )

        else:
            answer_text = "I could not find a matching retail question."

        st.success(f"🤖 {answer_text}")
        speak_text(answer_text, language)

if st.button("🔊 Listen to Dashboard Summary"):

    if language == "English":
        voice_text = (
            f"Welcome to the Retail Sales Intelligence Dashboard. "
            f"The highest predicted demand is for {highest_category} "
            f"during {highest_season}. "
            f"The expected quantity is {highest_quantity} units. "
            f"The highest customer activity is in {busiest_zone.replace('_', ' ')}. "
            f"The average customer activity is "
            f"{busiest_zone_value:.2f} customers."
        )

    elif language == "ಕನ್ನಡ (Kannada)":
        voice_text = (
            f"ರಿಟೇಲ್ ಸೇಲ್ಸ್ ಇಂಟೆಲಿಜೆನ್ಸ್ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್‌ಗೆ ಸ್ವಾಗತ. "
            f"{highest_season} ಸಮಯದಲ್ಲಿ {highest_category} ಉತ್ಪನ್ನಗಳಿಗೆ "
            f"ಹೆಚ್ಚಿನ ಬೇಡಿಕೆ ನಿರೀಕ್ಷಿಸಲಾಗಿದೆ. "
            f"ನಿರೀಕ್ಷಿತ ಪ್ರಮಾಣ {highest_quantity} ಘಟಕಗಳು. "
            f"ಹೆಚ್ಚಿನ ಗ್ರಾಹಕರ ಚಟುವಟಿಕೆ "
            f"{busiest_zone.replace('_', ' ')} ನಲ್ಲಿ ಕಂಡುಬಂದಿದೆ. "
            f"ಸರಾಸರಿ ಗ್ರಾಹಕರ ಚಟುವಟಿಕೆ "
            f"{busiest_zone_value:.2f} ಆಗಿದೆ."
        )

    else:
        voice_text = (
            f"रिटेल सेल्स इंटेलिजेंस डैशबोर्ड में आपका स्वागत है। "
            f"{highest_season} के दौरान {highest_category} की "
            f"सबसे अधिक मांग होने का अनुमान है। "
            f"अनुमानित मात्रा {highest_quantity} यूनिट है। "
            f"सबसे अधिक ग्राहक गतिविधि "
            f"{busiest_zone.replace('_', ' ')} में है। "
            f"औसत ग्राहक गतिविधि "
            f"{busiest_zone_value:.2f} है।"
        )

    speak_text(voice_text, language)
# ==============================
# Business Insights
# ==============================

st.subheader("💡 Business Insights")

# =================================
# Business Insight Messages
# =================================

if language == "English":

    demand_title = "📦 Highest Current Demand"
    demand_message = (
        f"{dynamic_highest_category} during {dynamic_highest_season}"
    )
    quantity_message = (
        f"Quantity sold: {dynamic_highest_quantity} units"
    )

    activity_title = "📍 Highest Customer Activity"
    activity_message = busiest_zone.replace("_", " ")
    activity_value = (
        f"Average customers: {busiest_zone_value:.2f}"
    )

    recommendation = (
        f"Consider maintaining sufficient stock of "
        f"{dynamic_highest_category} for the high-demand "
        f"{dynamic_highest_season} period in the current dataset."
    )

elif language == "ಕನ್ನಡ (Kannada)":

    demand_title = "📦 ಪ್ರಸ್ತುತ ಹೆಚ್ಚು ಬೇಡಿಕೆ"
    dynamic_kannada_season = kannada_seasons.get(
        dynamic_highest_season, dynamic_highest_season
    )
    dynamic_kannada_category = kannada_categories.get(
        dynamic_highest_category, dynamic_highest_category
    )
    demand_message = (
        f"{dynamic_kannada_season} ಸಮಯದಲ್ಲಿ {dynamic_kannada_category}"
    )
    quantity_message = (
        f"ಮಾರಾಟವಾದ ಪ್ರಮಾಣ: {dynamic_highest_quantity} ಘಟಕಗಳು"
    )

    activity_title = "📍 ಹೆಚ್ಚು ಗ್ರಾಹಕರ ಚಟುವಟಿಕೆ"
    activity_message = kannada_zone
    activity_value = (
        f"ಸರಾಸರಿ ಗ್ರಾಹಕರು: {busiest_zone_value:.2f}"
    )

    recommendation = (
        f"ಪ್ರಸ್ತುತ ಡೇಟಾದಲ್ಲಿ ಹೆಚ್ಚು ಬೇಡಿಕೆಯಿರುವ "
        f"{dynamic_kannada_category} ಉತ್ಪನ್ನಗಳನ್ನು "
        f"{dynamic_kannada_season} ಸಮಯದಲ್ಲಿ ಸಾಕಷ್ಟು ಸ್ಟಾಕ್‌ನಲ್ಲಿ "
        f"ಇಟ್ಟುಕೊಳ್ಳಲು ಪರಿಗಣಿಸಿ."
    )

else:

    demand_title = "📦 वर्तमान में सबसे अधिक मांग"
    dynamic_hindi_season = hindi_seasons.get(
        dynamic_highest_season, dynamic_highest_season
    )
    dynamic_hindi_category = hindi_categories.get(
        dynamic_highest_category, dynamic_highest_category
    )
    demand_message = (
        f"{dynamic_hindi_season} में {dynamic_hindi_category}"
    )
    quantity_message = (
        f"बिकी हुई मात्रा: {dynamic_highest_quantity} यूनिट"
    )

    activity_title = "📍 सबसे अधिक ग्राहक गतिविधि"
    activity_message = hindi_zone
    activity_value = (
        f"औसत ग्राहक: {busiest_zone_value:.2f}"
    )

    recommendation = (
        f"वर्तमान डेटा में अधिक मांग वाले "
        f"{dynamic_hindi_category} का "
        f"{dynamic_hindi_season} अवधि में पर्याप्त स्टॉक रखने "
        f"पर विचार करें।"
    )

# =================================
# Display Insights
# =================================

col1, col2 = st.columns(2)

with col1:

    st.info(
        f"**{demand_title}**\n\n"
        f"{demand_message}\n\n"
        f"{quantity_message}"
    )

with col2:

    st.info(
        f"**{activity_title}**\n\n"
        f"{activity_message}\n\n"
        f"{activity_value}"
    )

st.success(
    f"💡 **Business Recommendation:** {recommendation}"
)

# ==============================
# 🤖 AI Business Chatbot
# ==============================

st.divider()
st.subheader("🤖 AI Business Chatbot")
st.write(
    "Type a business question about the current filtered dataset. "
    "The chatbot uses the same multilingual semantic AI used by the voice assistant."
)

chatbot_question = st.text_input(
    "💬 Ask your business question",
    placeholder="Example: Which category has the highest profit?",
    key="business_chatbot_question"
)

chatbot_examples = [
    "What are the total sales?",
    "Which category has the highest profit?",
    "Which region has the highest sales?",
    "Which category sells the most?",
    "What is the profit margin?",
    "Which zone is busiest?",
    "Which season has the highest demand?",
    "Give me a business summary"
]

st.caption("Try: " + " • ".join(chatbot_examples[:5]))

if chatbot_question:
    chatbot_intent, chatbot_confidence = understand_intent(chatbot_question)

    st.info(
        f"🧠 Detected Intent: {chatbot_intent} | "
        f"Confidence: {chatbot_confidence:.2f}"
    )

    # Additional text-chat analysis for common comparison/business questions.
    chatbot_lower = chatbot_question.lower()
    chatbot_answer = None

    if "profit margin" in chatbot_lower:
        if language == "English":
            chatbot_answer = (
                f"The overall profit margin for the current filtered dataset is "
                f"{overall_profit_margin:.2f}%."
            )
        elif language == "ಕನ್ನಡ (Kannada)":
            chatbot_answer = (
                f"ಪ್ರಸ್ತುತ ಫಿಲ್ಟರ್ ಮಾಡಿದ ಡೇಟಾದ ಒಟ್ಟು ಲಾಭದ ಅಂಚು "
                f"{overall_profit_margin:.2f}% ಆಗಿದೆ."
            )
        else:
            chatbot_answer = (
                f"वर्तमान फ़िल्टर किए गए डेटासेट का कुल लाभ मार्जिन "
                f"{overall_profit_margin:.2f}% है।"
            )

    elif "compare" in chatbot_lower and "Category" in df.columns:
        matched_categories = [
            category for category in df["Category"].dropna().unique()
            if str(category).lower() in chatbot_lower
        ]
        if len(matched_categories) >= 2:
            comparison = (
                df[df["Category"].isin(matched_categories)]
                .groupby("Category", as_index=False)
                .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Quantity=("Quantity", "sum"))
            )
            chatbot_answer = "Category comparison:\n" + "\n".join(
                f"{row['Category']}: Sales ₹{row['Sales']:,.2f}, "
                f"Profit ₹{row['Profit']:,.2f}, Quantity {int(row['Quantity']):,} units"
                for _, row in comparison.iterrows()
            )

    elif ("highest sales region" in chatbot_lower or "region has the highest sales" in chatbot_lower) and "Region" in df.columns:
        region_sales_chat = (
            df.groupby("Region", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=False)
        )
        if not region_sales_chat.empty:
            top_region = region_sales_chat.iloc[0]
            chatbot_answer = (
                f"{top_region['Region']} has the highest sales in the current filtered dataset, "
                f"with ₹{top_region['Sales']:,.2f}."
            )

    if chatbot_answer is None and chatbot_confidence >= 0.45:
        # Reuse the voice assistant's answer logic for supported intents.
        if chatbot_intent == "TOTAL_SALES":
            chatbot_answer = f"The total sales are ₹{total_sales:,.2f}."
        elif chatbot_intent == "TOTAL_PROFIT":
            chatbot_answer = f"The total profit is ₹{total_profit:,.2f}."
        elif chatbot_intent == "TOTAL_QUANTITY":
            chatbot_answer = f"The total quantity sold is {total_quantity:,} units."
        elif chatbot_intent == "TOTAL_ORDERS":
            chatbot_answer = f"The total number of orders is {total_orders:,}."
        elif chatbot_intent == "AVERAGE_SALES":
            chatbot_answer = f"The average sales value per record is ₹{average_sales:,.2f}."
        elif chatbot_intent == "HIGHEST_PROFIT_CATEGORY":
            chatbot_answer = (
                f"{highest_profit_category} has the highest total profit, "
                f"with ₹{highest_profit_category_value:,.2f}."
            )
        elif chatbot_intent == "HIGHEST_DEMAND_SEASON":
            chatbot_answer = (
                f"The highest quantity sold in the current dataset is in "
                f"{dynamic_highest_season}, for {dynamic_highest_category}, "
                f"with {dynamic_highest_quantity} units."
            )
        elif chatbot_intent == "CUSTOMERS_ENTERED":
            chatbot_answer = f"{total_entered} customers entered the store."
        elif chatbot_intent == "CUSTOMERS_EXITED":
            chatbot_answer = f"{total_exited} customers exited the store."
        elif chatbot_intent == "CURRENT_OCCUPANCY":
            chatbot_answer = f"The latest recorded store occupancy is {current_occupancy} customers."
        elif chatbot_intent == "PEAK_OCCUPANCY":
            chatbot_answer = f"The peak store occupancy was {peak_occupancy} customers."
        elif chatbot_intent == "AVERAGE_OCCUPANCY":
            chatbot_answer = f"The average customer occupancy is {average_occupancy:.2f} customers."
        elif chatbot_intent == "BUSIEST_ZONE":
            chatbot_answer = (
                f"{busiest_zone} has the highest customer activity, "
                f"with an average of {busiest_zone_value:.2f} customers."
            )
        elif chatbot_intent == "LOWEST_ACTIVITY_ZONE":
            chatbot_answer = (
                f"{lowest_zone} has the lowest customer activity, "
                f"with an average of {lowest_zone_value:.2f} customers."
            )
        elif chatbot_intent == "AVERAGE_DWELL":
            chatbot_answer = f"The average customer dwell time is {average_dwell:.2f} seconds."
        elif chatbot_intent == "MAXIMUM_DWELL":
            chatbot_answer = f"The maximum customer dwell time is {maximum_dwell:.2f} seconds."
        elif chatbot_intent == "SALES_FORECAST":
            chatbot_answer = (
                f"The forecast contains 12 predicted values. The average predicted sales is "
                f"₹{average_forecast:,.2f}."
            )
        elif chatbot_intent == "BUSINESS_SUMMARY":
            chatbot_answer = (
                f"Business summary: sales ₹{total_sales:,.2f}, profit ₹{total_profit:,.2f}, "
                f"orders {total_orders:,}, quantity {total_quantity:,} units, "
                f"busiest zone {busiest_zone}, and highest current demand "
                f"for {dynamic_highest_category} during {dynamic_highest_season}."
            )
        elif chatbot_intent == "HIGHEST_DEMAND":
            chatbot_answer = (
                f"The highest predicted demand is for {highest_category} during "
                f"{highest_season}, with {highest_quantity} units."
            )

    if chatbot_answer is not None:
        st.success(f"🤖 {chatbot_answer}")
    else:
        st.warning(
            "I could not confidently answer that question. Try asking about sales, "
            "profit, quantity, orders, customers, zones, dwell time, demand, "
            "profit margin, forecast, or business summary."
        )

# ==============================
# Business Report Generator
# ==============================

st.divider()

st.subheader("📄 Business Report")

st.write(
    "Generate a business-friendly PDF report "
    "containing sales, seasonal demand, customer "
    "and store zone insights."
)

if st.button(
    "📄 Generate Business Report",
    use_container_width=True
):

    try:

        # Convert dashboard language
        # to the language expected by business_report.py

        report_language = language.replace(
            "ಕನ್ನಡ (Kannada)",
            "Kannada"
        ).replace(
            "हिन्दी (Hindi)",
            "Hindi"
        )

        result = subprocess.run(
            [
                sys.executable,
                "src/business_report.py",
                report_language
            ],
            capture_output=True,
            text=True,
            check=True
        )

        st.success(
            "Business report generated successfully!"
        )

        report_path = (
            "reports/retail_business_report.pdf"
        )

        with open(
            report_path,
            "rb"
        ) as report_file:

            report_data = report_file.read()

        st.download_button(
            label="⬇️ Download Business Report",
            data=report_data,
            file_name="retail_business_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    except subprocess.CalledProcessError as error:

        st.error(
            "Unable to generate the business report."
        )

        st.code(
            error.stderr
        )

# ==============================
# ✅ Project Feature Status
# ==============================

st.divider()
with st.expander("✅ Project Feature Checklist"):
    st.markdown(
        """
        - ✅ CSV Dataset Upload
        - ✅ Data Quality Check
        - ✅ Interactive Category / Region / Segment / Date Filters
        - ✅ Sales, Profit, Orders and Quantity KPIs
        - ✅ Sales and Profit Visualizations
        - ✅ Sales Trend Analysis
        - ✅ Top 10 Products Analysis
        - ✅ Seasonal Demand Analysis
        - ✅ Random Forest Sales Forecasting
        - ✅ Computer Vision Customer Entry / Exit Analytics
        - ✅ Occupancy Analytics
        - ✅ Customer Dwell Time Analytics
        - ✅ Store Zone Analytics
        - ✅ Profit Margin Analysis
        - ✅ AI Voice Assistant
        - ✅ English / Kannada / Hindi Support
        - ✅ AI Business Chatbot
        - ✅ Category / Region / Segment Comparison
        - ✅ Sales Anomaly Detection
        - ✅ Business Insights
        - ✅ Business Report PDF
        - ✅ Download Filtered Sales Data
        - ✅ Customer Behavior & Sales Context
        """
    )
