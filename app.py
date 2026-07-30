import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import the exact functions from your team's files
from data_loader import load_data
from preprocessing import clean_data1, clean_data2, merge
from visualization import plot_yearly_growth, plot_solar_vs_wind, plot_regional_distribution, plot_forecast_by_status
from evaluation import evaluate_forecast, evaluate_dual_forecasts

# Application page configuration
st.set_page_config(page_title="Saudi Renewable Energy AI", layout="wide")

st.title("Saudi Arabia Renewable Energy Forecasting Platform")
st.subheader("An interactive AI-driven MLOps dashboard analyzing and forecasting Saudi Arabia's Vision 2030 renewable energy targets.")

st.write("---")
# Initialize safe fallback variables to prevent NameError during tight filtering
gap_val = 0.0
has_installed_forecast = False
has_combined_forecast = False

# Load and clean the raw data using team's functions (Runs once to keep app fast)
try:
    data1_raw, data2_raw = load_data()
    d1_cleaned = clean_data1(data1_raw)
    d2_cleaned = clean_data2(data2_raw)
    df_merged = merge(d1_cleaned, d2_cleaned)
    
    # 🛠️ CRITICAL COLUMNS MATCHING FIX
    if 'Capacity (MW)' in df_merged.columns:
        df_merged['Capacity'] = df_merged['Capacity (MW)']
    if 'Capacity (MW)' in d1_cleaned.columns:
        d1_cleaned['Capacity'] = d1_cleaned['Capacity (MW)']
        
except Exception as e:
    st.error(f"Backend Pipeline Initialization Error: {str(e)}")
    st.stop()

# SIDEBAR CONTROLS: Dynamic User Selection (Interactive Filters)
st.sidebar.header("🕹️ Interactive User Controls")
st.sidebar.write("Select filters to customize the data preview and KPIs:")


# Get unique cities and years
available_cities = sorted(df_merged["City"].dropna().unique().tolist())
available_years = sorted(
    df_merged["Year"].dropna().astype(int).unique().tolist()
)

# Add one clear All option
city_options = ["All Cities"] + available_cities
year_options = ["All Years"] + available_years

selected_cities = st.sidebar.multiselect(
    "📍 Select City / Region:",
    city_options,
    default=["All Cities"]
)

selected_years = st.sidebar.multiselect(
    "📅 Select Target Years:",
    year_options,
    default=["All Years"]
)
if "All Cities" in selected_cities and len(selected_cities) > 1:
    selected_cities = [city for city in selected_cities if city != "All Cities"]

if "All Years" in selected_years and len(selected_years) > 1:
    selected_years = [year for year in selected_years if year != "All Years"]
# Apply User Filters dynamically to the dataframe
df_filtered = df_merged.copy()
d1_filtered = d1_cleaned.copy()

# Apply city filter only when All Cities is not selected
if selected_cities and "All Cities" not in selected_cities:
    df_filtered = df_filtered[
        df_filtered["City"].isin(selected_cities)
    ]
    d1_filtered = d1_filtered[
        d1_filtered["City"].isin(selected_cities)
    ]

# Apply year filter only when All Years is not selected
if selected_years and "All Years" not in selected_years:
    df_filtered = df_filtered[
        df_filtered["Year"].astype(int).isin(selected_years)
    ]
    d1_filtered = d1_filtered[
        d1_filtered["Year"].astype(int).isin(selected_years)
    ]

# Calculate real KPI metrics dynamically based on USER SELECTION
total_projects = len(df_filtered)
total_current_capacity = df_filtered['Capacity'].sum() if total_projects > 0 else 0
avg_capacity = df_filtered['Capacity'].mean() if total_projects > 0 else 0

# Display real system KPI metrics calculated from filtered data
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="📊 Tracked Projects (Filtered)", value=f"{total_projects} Projects")
with col2:
    st.metric(label="⚡ Filtered System Capacity", value=f"{int(total_current_capacity):,} MW")
with col3:
    st.metric(label="📐 Average Project Capacity", value=f"{avg_capacity:.2f} MW" if total_projects > 0 else "0.00 MW")
    
st.write("---")

# Grid Layout for the Core Team Analytics & Global AI Forecasts
st.subheader("📊 System Visualizations & AI Forecasts")
# Safely format multiple selections into a readable string separating them with commas
city_display = (
    "All Cities"
    if not selected_cities or "All Cities" in selected_cities
    else ", ".join(selected_cities)
)

year_display = (
    "All Years"
    if not selected_years or "All Years" in selected_years
    else ", ".join(map(str, selected_years))
)

st.write(
    f"Displaying renewable energy statistics and AI forecasts "
    f"for **{city_display}** across **{year_display}**."
)

# Making sure that there is data after filtering
if total_projects > 0:
    # First row of charts: Growth and Solar vs Wind
    c1, c2 = st.columns(2)
    with c1:
        st.write("### Saudi Arabia – Yearly Renewable Energy Growth (Installed Projects)")
        
        # Take the filtered dataframe and extract rows that are strictly 'Installed'
        d1_dynamic = df_filtered[df_filtered['Installed / Planned'] == 'Installed'].copy() if 'Installed / Planned' in df_filtered.columns else df_filtered.copy()

        # Check if this dynamic slice actually contains any rows to plot
        if len(d1_dynamic) > 0 and d1_dynamic['Capacity'].sum() > 0:
           
            # Combine multi-project rows for the same year to fix duplicate X-axis labels
            cleaned_input = d1_dynamic.groupby('Year', as_index=False)['Capacity'].sum()
            cleaned_input['Year'] = cleaned_input['Year'].astype(int)
            cleaned_input['Installed / Planned'] = 'Installed' # Keep the column intact for the backend function

            # Clear the matplotlib memory to prevent chart overlapping
            plt.clf()
            
            # Create a new figure
            fig1 = plt.figure(figsize=(8, 4.5))
            
            # Call the original function with the perfectly grouped data
            plot_yearly_growth(cleaned_input)
            st.pyplot(plt.gcf())
            plt.close(fig1)
            st.info("**Chart Logic & Insight:** This visualization filters the dataset to include only operational (Installed) projects, grouping the filtered data by Year to aggregate and calculate the total sum of annual capacity additions (MW).")
        else:
            # Show a clean warning instead of an empty/blank plot grid
            st.warning(f"📊 **No Active Installed Projects for {city_display} ({year_display}):** \n\n"
                       f"There are currently no operational 'Installed' capacity entries registered in the dataset for this selection. "
                       f"Please adjust the filters (e.g., select All Years or 2024) to view the region's active growth metrics.")
        
    with c2:
        st.write("### Solar vs Wind Energy Comparison")
        fig2 = plt.figure(figsize=(8, 4.5))
        plot_solar_vs_wind(df_filtered)
        st.pyplot(plt.gcf())
        plt.close(fig2)
        st.info("**Chart Logic & Insight:** This multi-plot executes a comparative analysis by grouping data across both project status and energy types. The left bar chart tracks capacity metrics, while the right pie chart illustrates the total energy mix share proportions.")

    # Second row of charts: Regional and Vision 2030 AI Forecast
    c3, c4 = st.columns(2)
    with c3:
        st.write("### Regional Renewable Energy Distribution (Installed vs Planned)")
        fig3 = plt.figure(figsize=(8, 4.5))

        # To handle citites that doesn't have both installed and planned data
        try:
            plot_regional_distribution(df_filtered)
            st.pyplot(plt.gcf())
            st.info("**Chart Logic & Insight:** This horizontal stacked bar chart displays energy capacity across Saudi cities. To focus strictly on specific individual regions, residual 'Multi-city' entries are safely excluded, sorting the remaining locations by their operational assets.")
        except KeyError as ke:
            # Displays a clean, localized summary metric box when a column like 'Installed' is mathematically missing
            st.info(f"📊 **Localized Capacity Breakdown for {city_display} ({year_display}):**\n\n"
                    f"Currently, there are no 'Installed' baseline projects active for this dynamic selection. "
                    f"The total combined pipeline consists strictly of **{int(total_current_capacity):,} MW** from upcoming **Planned** infrastructure assets.")
        except Exception as e:
            st.warning("Unable to render comparative regional distribution for this tight data slice.")
        
        plt.close(fig3)
        
    with c4:
        st.write("### Future Renewable Energy Capacity Forecast – Installed Baseline")
        fig4 = plt.figure(figsize=(8, 4.5))

        # To make sure that there are more than two points (two different years) to calculate the linear prediction
        if len(df_filtered['Year'].dropna().unique()) > 1:
            try:
                future_2030, vision_target, gap_val, r2_val, slope_val = plot_forecast_by_status(df_filtered, status_type='Installed', forecast_until=2030)
                st.pyplot(plt.gcf())
                st.info("**Forecast Interpretation (Installed Track):** The model uses historical renewable energy growth to estimate future cumulative capacity. This forecast is approximate because it is based on limited historical data and a linear trend. Using installed projects only, the 2030 capacity may be about below the Vision 2030 target.")
                has_installed_forecast = True
            except Exception as forecast_err:
                st.error("⚠️ Cannot calculate linear forecast: The mathematical data matrix is singular (insufficient variation in historical points).")
                has_installed_forecast = False
        else:
            st.warning(f"📊 **Linear Forecasting Disabled for {city_display}:** \n\n"
                       f"A Linear Regression model mathematically requires at least **2 different years** to establish a trend line ($Y = mX + c$). "
                       f"The selected filter only contains data for 1 year, making it impossible to calculate a slope.")
            has_installed_forecast = False
        plt.close(fig4)

    
    # 5TH GRAPH
    st.write("---")
    st.write("### Future Renewable Energy Capacity Forecast – Combined (Installed + Planned)")
    fig5_bottom = plt.figure(figsize=(10, 5))

    # To make sure there are more than two years
    if len(df_filtered['Year'].dropna().unique()) > 1:
        try:
            p_2030, p_target, p_gap, p_r2, p_slope = plot_forecast_by_status(df_filtered, status_type='Planned', forecast_until=2030)
            st.pyplot(plt.gcf())
            st.info("**Forecast Interpretation (Combined Track):** This predictive model evaluates the combined pipeline by running historical data alongside upcoming planned projects through a cumulative sum. After adding planned assets, the expected 2030 gap decreases significantly to help fully align with the national target line.")
            has_combined_forecast = True
        except Exception as combined_err:
            st.error("⚠️ Cannot calculate combined linear forecast due to a mathematical alignment error in the data pipeline.")
            has_combined_forecast = False
    else:
        st.warning(f"📊 **Combined Forecasting Disabled for {city_display}:** \n\n"
                   f"Insufficient data density. To plot a trajectory toward 2030, the model requires historical data points across multiple years to perform cumulative regression mapping.")
        has_combined_forecast = False
    plt.close(fig5_bottom)

    
    # User-Centric AI Model Evaluation Report Section
    st.write("---")
    st.subheader("🎯 Automated Vision 2030 Alignment Report (Installed + Planned)")

    # The report only appears if the prediction succeeds
    if has_combined_forecast:
        # call the evaluate function
        metrics = evaluate_forecast(p_2030, p_target, p_gap, p_r2)
        if has_installed_forecast:
            inst_metrics = (future_2030, vision_target, gap_val, r2_val, slope_val)
            plan_metrics = (p_2030, p_target, p_gap, p_r2, p_slope)
            evaluate_dual_forecasts(inst_metrics, plan_metrics)
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            st.success(f"**Model Trend Fit (R² Score):** {metrics['trend_fit']:.4f}")
        with ec2:
            if metrics['gap'] > 0:
                st.warning(f"**Projected Vision Gap:** {int(metrics['gap']):,} MW remaining")
            else:
                st.success("**Target Met or Exceeded for this selection!**")
        with ec3:
            st.metric(label="Target Achievement Rate", value=f"{metrics['achievement_rate']:.2f}%")

        # Clear and simple forecasting explanation for the user
        st.markdown("### 💡 Forecast Interpretation")
        st.info(f"""
        * The model uses historical renewable energy growth to estimate future cumulative capacity.
        * This forecast is approximate because it is based on limited historical data and a linear trend.
        * Using installed projects only, the 2030 capacity may be about **{gap_val:,.0f} MW** below the Vision 2030 target.
        * After adding planned projects, the expected 2030 gap decreases to about **{p_gap:,.0f} MW**.
        * Planned projects may reduce the gap by about **{(gap_val - p_gap):,.0f} MW**.
        """)
    else:
        st.info("💡 **Alignment Report Summary Unavailable:** \n\n"
                "The target alignment evaluation metrics cannot be displayed because the predictive model requires a broader historical time-series baseline to compute standard R² and Achievement rates for this localized region.")

else:
    st.warning(f"No projects found in the dataset for the selected combination. Please adjust the filters to view visualizations.")

# Dynamic Interacted Dataframe Display at the bottom
st.write("---")
st.subheader(f"📋 Dataset Preview")
if total_projects > 0:
    st.dataframe(df_filtered, use_container_width=True)
else:
    st.warning(f"No projects found in the dataset for the selected combination.")
