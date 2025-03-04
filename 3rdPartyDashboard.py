import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
import numpy as np
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Order and Subscription Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS (unchanged from previous code)
st.markdown("""
<style>
    .main {
        background-color: #f5f5f7;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .card {
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f7f7f7;
        border-radius: 5px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #1f78b4;
    }
    .metric-title, .metric-label {
        font-size: 14px;
        color: #666;
    }
    .section-title {
        color: #333;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .filter-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .dashboard-title {
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .stPlotlyChart {
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-radius: 5px;
        padding: 5px;
    }
    .subs-filters .stSelectbox, .subs-filters .stMultiSelect {
        background-color: white;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-family: sans-serif;
        padding: 5px;
    }
    .subs-filters .stMultiSelect div[data-baseweb="tag"] {
        background-color: #e0e0e0;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Data loading functions
@st.cache_data
def load_orders_data(file):
    df = pd.read_excel(file, sheet_name="Consolidated Order Summary")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    latest_orders = df.sort_values(['OrderId', 'Date']).groupby('OrderId').last().reset_index()
    return df, latest_orders

@st.cache_data
def load_subscriptions_data(file):
    df = pd.read_excel(file, sheet_name="Manage Subscription")
    date_formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]
    for fmt in date_formats:
        try:
            df["Date"] = pd.to_datetime(df["Date"], format=fmt, errors='coerce')
            if not df["Date"].isna().all():
                break
        except ValueError:
            continue
    df = df.dropna(subset=["Date"])
    for col in ["Transaction Type", "Order Status", "Test Order"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    df = df.sort_index().groupby("OrderId", as_index=False).last()
    return df

# Helper functions
def get_month_date_range():
    today = datetime.now()
    first_day = today.replace(day=1)
    _, last_day = calendar.monthrange(today.year, today.month)
    last_date = today.replace(day=last_day)
    return first_day, last_date

def get_all_date_range(df):
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    return min_date, max_date

# Display functions
def display_orders_analysis(df_orders, latest_orders):
    st.markdown("<div class='section-title'>📅 Filter Options</div>", unsafe_allow_html=True)
    st.markdown("<div class='filter-container subs-filters'>", unsafe_allow_html=True)

    # Filters on a single line
    col1, col2, col3 = st.columns(3)
    with col1:
        date_filter_option = st.selectbox(
            "Select timeframe",
            ["All", "Current Month", "Last 30 Days", "Last 7 Days", "Custom"],
            key="order_analysis_date_filter"
        )
    with col2:
        status_options = ["All"] + sorted(latest_orders["Order Status"].unique().tolist())
        selected_status = st.selectbox("Filter by Order Status", status_options, key="orders_status")
    with col3:
        order_type_options = ["All", "Real-Time", "Test"]
        selected_order_type = st.selectbox("Filter by Order Type", order_type_options, key="orders_type")

    # Custom date inputs below if selected
    today = datetime.now()
    with st.expander("Custom Date Range", expanded=(date_filter_option == "Custom")):
        col1, col2 = st.columns(2)
        with col1:
            start_date_custom = st.date_input("Start date", today - timedelta(days=30), key="orders_start")
        with col2:
            end_date_custom = st.date_input("End date", today, key="orders_end")
    
    if date_filter_option == "Custom":
        start_date = datetime.combine(start_date_custom, datetime.min.time())
        end_date = datetime.combine(end_date_custom, datetime.max.time())
    else:
        if date_filter_option == "All":
            start_date, end_date = get_all_date_range(df_orders)
        elif date_filter_option == "Current Month":
            start_date, end_date = get_month_date_range()
        elif date_filter_option == "Last 30 Days":
            start_date = today - timedelta(days=30)
            end_date = today
        elif date_filter_option == "Last 7 Days":
            start_date = today - timedelta(days=7)
            end_date = today

    st.markdown("</div>", unsafe_allow_html=True)

    # Filter data
    filtered_df = df_orders[(df_orders['Date'] >= start_date) & (df_orders['Date'] <= end_date)]
    filtered_latest = latest_orders[(latest_orders['Date'] >= start_date) & (latest_orders['Date'] <= end_date)]

    if selected_status != "All":
        filtered_latest = filtered_latest[filtered_latest["Order Status"] == selected_status]
    if selected_order_type == "Real-Time":
        filtered_latest = filtered_latest[(filtered_latest["Test Order"] == "No") | (filtered_latest["Test Order"].isna())]
    elif selected_order_type == "Test":
        filtered_latest = filtered_latest[filtered_latest["Test Order"] == "Yes"]

    # Metrics (unchanged)
    total_orders = len(filtered_latest)
    real_time_orders = len(filtered_latest[(filtered_latest["Test Order"] == "No") | (filtered_latest["Test Order"].isna())])
    test_orders = len(filtered_latest[filtered_latest["Test Order"] == "Yes"])
    successful_real_time = len(filtered_latest[
        ((filtered_latest["Test Order"] == "No") | (filtered_latest["Test Order"].isna())) &
        (filtered_latest["Order Status"] == "Successful")
    ])
    successful_test = len(filtered_latest[
        (filtered_latest["Test Order"] == "Yes") &
        (filtered_latest["Order Status"] == "Successful")
    ])
    cancelled_orders = len(filtered_latest[filtered_latest["Order Status"] == "Cancelled"])
    pending_orders = len(filtered_latest[
        (filtered_latest["Order Status"] == "Pending") | (filtered_latest["Order Status"].isna())
    ])

    st.markdown("<div class='section-title'>📌 Order Status Summary</div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_orders}</div><div class='metric-title'>Total Orders</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{real_time_orders}</div><div class='metric-title'>Real-Time Orders</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{test_orders}</div><div class='metric-title'>Test Orders</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{successful_real_time}</div><div class='metric-title'>Successful Real-Time</div></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{successful_test}</div><div class='metric-title'>Successful Test Orders</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{cancelled_orders}</div><div class='metric-title'>Cancelled Orders</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{pending_orders}</div><div class='metric-title'>Pending Orders</div></div>", unsafe_allow_html=True)

    # Visualizations (unchanged)
    st.markdown("<div class='section-title'>📈 Data Visualization</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        status_counts = filtered_latest['Order Status'].value_counts().reset_index()
        status_counts.columns = ['Order Status', 'Count']
        fig = px.pie(status_counts, values='Count', names='Order Status', title='Order Status Distribution', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        test_real_data = pd.DataFrame({'Type': ['Real-Time', 'Test'], 'Count': [real_time_orders, test_orders]})
        fig = px.bar(test_real_data, x='Type', y='Count', color='Type', title='Real-Time vs Test Orders', color_discrete_map={'Real-Time': '#72b7b2', 'Test': '#e377c2'})
        st.plotly_chart(fig, use_container_width=True)
    time_series = filtered_df.groupby(pd.Grouper(key='Date', freq='D')).size().reset_index(name='Orders')
    time_series['Date'] = time_series['Date'].dt.date
    fig = px.line(time_series, x='Date', y='Orders', title='Orders Over Time', markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # Detail view (unchanged)
    st.markdown("<div class='section-title'>🔍 Order Details</div>", unsafe_allow_html=True)
    columns_to_show = ['Date', 'OrderId', 'Transaction Type', 'Test Order', 'Comments', 'Order Status']
    st.dataframe(filtered_latest[columns_to_show], use_container_width=True)

    # Insights (unchanged)
    st.markdown("<div class='section-title'>💡 Insights</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        success_rate = (len(filtered_latest[filtered_latest["Order Status"] == "Successful"]) / total_orders * 100) if total_orders > 0 else 0
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=success_rate,
            title={'text': "Order Success Rate"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#1f78b4"}, 'steps': [
                {'range': [0, 33], 'color': "#ffcccb"}, {'range': [33, 66], 'color': "#ffffcc"}, {'range': [66, 100], 'color': "#ccffcc"}
            ]}
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    # Download (unchanged)
    st.markdown("<div class='section-title'>📥 Download Data</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        csv = filtered_latest.to_csv(index=False).encode('utf-8')
        st.download_button("Download as CSV", csv, "filtered_order_data.csv", "text/csv", key='orders_download_csv')
    with col2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            filtered_latest.to_excel(writer, index=False, sheet_name="Filtered Data")
        excel_data = output.getvalue()
        st.download_button("Download as Excel", excel_data, "filtered_order_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key='orders_download_excel')

def display_subscriptions_analysis(df_subscriptions):
    st.markdown("<div class='section-title'>📅 Filter Options</div>", unsafe_allow_html=True)
    st.markdown("<div class='filter-container subs-filters'>", unsafe_allow_html=True)

    # Filters on a single line
    col1, col2, col3 = st.columns(3)
    with col1:
        date_filter_option = st.selectbox(
            "Select timeframe",
            ["All", "Current Month", "Last 7 Days", "Custom"],
            key="manage_subscription_date_filter"
        )
    with col2:
        transaction_types = ["Add A Line", "Cancel Subscription", "Update A License"]
        selected_transaction_types = st.selectbox(
            "Filter by Transaction Type",
            ["All"] + transaction_types,
            key="subs_transaction"
        )
    with col3:
        order_types = st.selectbox(
            "Filter by Order Type",
            ["All", "Real-Time", "Test"],
            key="subs_order_type"
        )

    # Custom date inputs shown only when "Custom" is selected
    today = datetime.now()
    
    with st.expander("Custom Date Range", expanded=(date_filter_option == "Custom")):
        col1, col2 = st.columns(2)
        with col1:
            start_date_custom = st.date_input("Start date", today - timedelta(days=30), key="subscription_start")
        with col2:
            end_date_custom = st.date_input("End date", today, key="subscription_end")
    
    # Ensure Date column is datetime
    df_subscriptions['Date'] = pd.to_datetime(df_subscriptions['Date'], errors='coerce')
    
    # Date filtering logic
    filtered_df = df_subscriptions.copy()
    if date_filter_option == "Current Month":
        first_day, last_date = get_month_date_range()
        filtered_df = filtered_df[
            (filtered_df['Date'].dt.date >= first_day) & 
            (filtered_df['Date'].dt.date <= last_date)
        ]
    elif date_filter_option == "Last 7 Days":
        seven_days_ago = today - timedelta(days=7)
        filtered_df = filtered_df[filtered_df['Date'] >= pd.Timestamp(seven_days_ago)]
    elif date_filter_option == "Custom":
        start_date = datetime.combine(start_date_custom, datetime.min.time())
        end_date = datetime.combine(end_date_custom, datetime.max.time())
        filtered_df = filtered_df[
            (filtered_df['Date'] >= pd.Timestamp(start_date)) & 
            (filtered_df['Date'] <= pd.Timestamp(end_date))
        ]

    st.markdown("</div>", unsafe_allow_html=True)

    # Apply additional filters
    if selected_transaction_types != "All":
        filtered_df = filtered_df[filtered_df["Transaction Type"] == selected_transaction_types]
    
    # Order type filtering
    if order_types == "Real-Time":
        filtered_df = filtered_df[
            (filtered_df["Test Order"].isna()) | 
            (filtered_df["Test Order"].str.lower() == "no") | 
            (filtered_df["Test Order"] == "")
        ]
    elif order_types == "Test":
        filtered_df = filtered_df[filtered_df["Test Order"].str.lower() == "yes"]

    # Metrics calculation
    def calculate_metrics(df):
        # Prepare metrics dictionary
        metrics = {
            "Total Real-Time Orders": 0,
            "Successful Orders": 0,
            "Add-a-line Orders": 0,
            "Update a License Orders": 0,
            "Cancel Subscription Orders": 0,
            "Failed Orders": 0,
            "Test Orders": 0
        }
        
        # Real-Time Orders (excluding Test Orders)
        real_time_df = df[df["Test Order"].str.lower() != "yes"]
        
        # Aggregate by OrderId to get unique orders
        unique_orders = real_time_df.groupby('OrderId').first().reset_index()
        metrics["Total Real-Time Orders"] = len(unique_orders)
        
        # Successful Orders by Transaction Type
        successful_df = real_time_df[real_time_df["Order Status"] == "Successful"]
        successful_unique = successful_df.groupby('OrderId').first().reset_index()
        
        metrics["Add-a-line Orders"] = len(
            successful_unique[successful_unique["Transaction Type"] == "Add A Line"]
        )
        metrics["Update a License Orders"] = len(
            successful_unique[successful_unique["Transaction Type"] == "Update A License"]
        )
        metrics["Cancel Subscription Orders"] = len(
            successful_unique[successful_unique["Transaction Type"] == "Cancel Subscription"]
        )
        
        metrics["Successful Orders"] = (
            metrics["Add-a-line Orders"] + 
            metrics["Update a License Orders"] + 
            metrics["Cancel Subscription Orders"]
        )
        
        # Failed Orders
        failed_unique = real_time_df[real_time_df["Order Status"] == "Failed"].groupby('OrderId').first().reset_index()
        metrics["Failed Orders"] = len(failed_unique)
        
        # Test Orders
        test_df = df[df["Test Order"].str.lower() == "yes"]
        test_unique = test_df.groupby('OrderId').first().reset_index()
        metrics["Test Orders"] = len(test_unique)
        
        return metrics

    # Calculate metrics
    metrics = calculate_metrics(filtered_df)

    # Metrics Display
    st.markdown("<div class='section-title'>📌 Key Metrics</div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics['Total Real-Time Orders']}</div><div class='metric-title'>Total Real-Time Orders</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics['Successful Orders']}</div><div class='metric-title'>Successful Orders</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics['Add-a-line Orders']}</div><div class='metric-title'>Add-a-line Orders</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics['Failed Orders']}</div><div class='metric-title'>Failed Orders</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics['Update a License Orders']}</div><div class='metric-title'>Update a License Orders</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics['Test Orders']}</div><div class='metric-title'>Test Orders</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics['Cancel Subscription Orders']}</div><div class='metric-title'>Cancel Subscription Orders</div></div>", unsafe_allow_html=True)
        success_rate = (metrics["Successful Orders"] / metrics["Total Real-Time Orders"] * 100) if metrics["Total Real-Time Orders"] > 0 else 0
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{success_rate:.1f}%</div><div class='metric-title'>Success Rate</div></div>", unsafe_allow_html=True)

    # Visualizations (unchanged)
    st.markdown("<div class='section-title'>📈 Analytics</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        df_time = filtered_df.copy()
        df_time["Date"] = pd.to_datetime(df_time["Date"]).dt.date
        time_data = df_time.groupby(["Date", "Transaction Type"]).size().reset_index(name="Count")
        fig = px.line(time_data, x="Date", y="Count", color="Transaction Type", title="Transactions Over Time")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df_daily = filtered_df.copy()
        df_daily["Date"] = pd.to_datetime(df_daily["Date"]).dt.date
        success_by_day = df_daily.groupby(["Date", "Order Status"]).size().unstack(fill_value=0)
        fig = go.Figure()
        if not success_by_day.empty:
            if "Successful" not in success_by_day.columns:
                success_by_day["Successful"] = 0
            if "Failed" not in success_by_day.columns:
                success_by_day["Failed"] = 0
            success_by_day["Total"] = success_by_day["Successful"] + success_by_day["Failed"]
            success_by_day["Success Rate"] = success_by_day.apply(
                lambda row: (row["Successful"] / row["Total"] * 100) if row["Total"] > 0 else 0, axis=1
            )
            success_by_day = success_by_day.reset_index()
            fig.add_trace(go.Scatter(x=success_by_day["Date"], y=success_by_day["Success Rate"], mode="lines+markers", line=dict(color="#26A69A")))
            fig.update_layout(title="Daily Success Rate", yaxis=dict(range=[0, 105]))
        else:
            fig.update_layout(title="Daily Success Rate (No Data)", annotations=[{'text': "No data", 'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': 0.5}])
        st.plotly_chart(fig, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        status_counts = filtered_df.groupby("Order Status").size().reset_index(name="Count")
        fig = px.pie(status_counts, values="Count", names="Order Status", title="Order Status Distribution", hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df_test = filtered_df.copy()
        df_test["Order Type"] = df_test["Test Order"].apply(lambda x: "Test" if x == "Yes" else "Real-Time")
        type_counts = df_test.groupby("Order Type").size().reset_index(name="Count")
        fig = px.pie(type_counts, values="Count", names="Order Type", title="Test vs Real-Time", hole=0.6, color_discrete_map={"Real-Time": "#42A5F5", "Test": "#FFCA28"})
        st.plotly_chart(fig, use_container_width=True)

    # Data table with pagination
    st.markdown("<div class='section-title'>📋 Filtered Data</div>", unsafe_allow_html=True)
    display_columns = ["Date", "OrderId", "Transaction Type", "Order Status", "Comments", "Test Order"]
    display_df = filtered_df[display_columns].copy()
    items_per_page = 10
    total_items = len(display_df)
    total_pages = (total_items // items_per_page) + (1 if total_items % items_per_page > 0 else 0)
    if "subs_page" not in st.session_state:
        st.session_state.subs_page = 1
    start_idx = (st.session_state.subs_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    if total_pages > 0:
        st.dataframe(display_df.iloc[start_idx:end_idx].reset_index(drop=True), use_container_width=True)
        st.text(f"Showing {start_idx + 1} to {end_idx} of {total_items} entries")
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("Back", key="subs_back", disabled=(st.session_state.subs_page == 1)):
                st.session_state.subs_page -= 1
                st.rerun()
        with col2:
            if st.button("Next", key="subs_next", disabled=(st.session_state.subs_page == total_pages)):
                st.session_state.subs_page += 1
                st.rerun()
        with col3:
            st.write(f"Page {st.session_state.subs_page} of {total_pages}")
    else:
        st.info("No data to display")

    # Download options
    st.markdown("<div class='section-title'>📥 Download Data</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download as CSV", key="subs_csv"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(label="Click to Download CSV", data=csv, file_name="filtered_subscription_data.csv", mime="text/csv", key="subs_csv_download")
    with col2:
        if st.button("Download as Excel", key="subs_excel"):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name="Filtered Data")
            excel_data = output.getvalue()
            st.download_button(label="Click to Download Excel", data=excel_data, file_name="filtered_subscription_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="subs_excel_download")

# Page selection function
def page_selection():
    st.sidebar.title("Select Dashboard")
    page = st.sidebar.radio("Go to", ["Home", "Order Analysis", "Manage Subscription"])
    return page

# Home page
def home_page():
    st.title("📊 Order and Subscription Analysis Dashboard")
    st.write("Welcome to the comprehensive analysis dashboard.")
    st.write("Please select a dashboard from the sidebar to get started.")

# Main function
def main():
    page = page_selection()
    
    with st.sidebar:
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        with st.spinner('Loading and processing data...'):
            try:
                df_orders, latest_orders = load_orders_data(uploaded_file)
                df_subscriptions = load_subscriptions_data(uploaded_file)
                
                if page == "Home":
                    home_page()
                elif page == "Order Analysis":
                    st.title("📊 Order Analysis Dashboard")
                    display_orders_analysis(df_orders, latest_orders)
                elif page == "Manage Subscription":
                    st.title("🔄 Manage Subscription Dashboard")
                    display_subscriptions_analysis(df_subscriptions)
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    else:
        if page == "Home":
            home_page()
        else:
            st.info("Please upload an Excel file to view the dashboard")

if __name__ == "__main__":
    main()