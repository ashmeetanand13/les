import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Athlete Performance Dashboard - AOS",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to match the HTML design
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .risk-green { background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; }
    .risk-yellow { background-color: #FFC107; color: black; padding: 10px; border-radius: 5px; }
    .risk-orange { background-color: #FF9800; color: white; padding: 10px; border-radius: 5px; }
    .risk-red { background-color: #F44336; color: white; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif file_path and os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        return None
    
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    return df

# Sidebar
st.sidebar.markdown("## 🏃‍♂️ AOS")
st.sidebar.markdown("### Athlete Operating System")
st.sidebar.markdown("---")

# CSV File Upload
st.sidebar.markdown("### 📁 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

# Try to load data
default_path = '/mnt/user-data/uploads/athlete_data.csv'
if uploaded_file is not None:
    df = load_data(uploaded_file=uploaded_file)
    st.sidebar.success("✅ Using uploaded file")
elif os.path.exists(default_path):
    df = load_data(file_path=default_path)
    st.sidebar.info("📂 Using default file")
else:
    df = None
    st.sidebar.warning("⚠️ Please upload a CSV file")

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Team View", "Athlete Profile", "Injury Impact", "Wellness", "Workload", "Analytics"],
    index=1  # Default to Athlete Profile
)

# Date info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"📅 {datetime.now().strftime('%B %d, %Y')}")

# Main Content - Athlete Profile
if page == "Athlete Profile":
    
    if df is None:
        st.error("⚠️ No data available. Please upload a CSV file using the sidebar.")
    else:
        # Filters at the top
        col1, col2, col3 = st.columns([2, 2, 8])
        
        with col1:
            # Athlete filter
            athletes = df['Person'].unique()
            selected_athlete = st.selectbox("Select Athlete", athletes, index=0)
        
        with col2:
            # Date range filter
            date_range = st.date_input(
                "Date Range",
                value=(df['Date'].min(), df['Date'].max()),
                min_value=df['Date'].min(),
                max_value=df['Date'].max()
            )
            
            # Handle both single date and date range
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = end_date = date_range[0] if isinstance(date_range, tuple) else date_range
        
        # Filter data
        athlete_df = df[
            (df['Person'] == selected_athlete) & 
            (df['Date'] >= pd.Timestamp(start_date)) & 
            (df['Date'] <= pd.Timestamp(end_date))
        ].sort_values('Date')
        
        # Get latest data for the athlete
        if len(athlete_df) > 0:
            latest_data = athlete_df.iloc[-1]
            
            # Header with Athlete Info
            st.markdown(f"# 👤 {selected_athlete}")
            
            col1, col2, col3, col4 = st.columns([3, 3, 3, 3])
            with col1:
                st.metric("Last Activity", latest_data['Date'].strftime('%B %d, %Y'))
            with col2:
                st.metric("Total Sessions", len(athlete_df))
            with col3:
                st.metric("Current Readiness", latest_data.get('COMPOSITE RISK', 'N/A'))
            with col4:
                risk_color = latest_data.get('COMPOSITE RISK', 'GREEN')
                color_class = f"risk-{risk_color.lower()}"
                st.markdown(f'<div class="{color_class}">{risk_color}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # COMPOSITE RISK - PRIMARY METRIC
            st.markdown("## 🎯 COMPOSITE RISK STATUS")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                risk_status = latest_data.get('COMPOSITE RISK', 'GREEN')
                if risk_status == 'GREEN':
                    st.success(f"**Status:** {risk_status}")
                elif risk_status == 'YELLOW':
                    st.warning(f"**Status:** {risk_status}")
                elif risk_status == 'ORANGE':
                    st.warning(f"**Status:** {risk_status}")
                else:
                    st.error(f"**Status:** {risk_status}")
                    
            with col2:
                st.metric("Risk Multiplier", f"{latest_data.get('RISK MULTIPLIER', 1.0)}x")
                
            with col3:
                training_adj = latest_data.get('TRAINING ADJUSTMENT', 'NO ADJUSTMENT')
                st.metric("Training Adjustment", training_adj)
                
            with col4:
                workload_risk = latest_data.get('WORKLOAD W/ RISK MULTIPLIER', 0.0)
                st.metric("Workload w/ Risk", f"{workload_risk:.2f}")
            
            st.markdown("---")
            
            # Today's Monitoring Loop
            st.markdown("## 📊 Today's Monitoring Loop")
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["Workload (Dose)", "Response & Wellness", "Risk Flags", "Training Recommendation"])
            
            with tab1:
                # Workload metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    volume_score = latest_data.get('VOLUME STEN', 0)
                    st.metric("Volume", f"{volume_score:.1f}")
                    st.progress(min(volume_score/10, 1.0))
                    
                with col2:
                    intensity_score = latest_data.get('INTENSITY STEN', 0)
                    st.metric("Intensity", f"{intensity_score:.1f}")
                    st.progress(min(intensity_score/10, 1.0))
                    
                with col3:
                    symmetry_score = latest_data.get('SYMMETRY STEN', 0)
                    st.metric("Symmetry", f"{symmetry_score:.1f}")
                    st.progress(min(symmetry_score/10, 1.0))
                    
                with col4:
                    performance_score = latest_data.get('PERFORMANCE STEN', 0)
                    st.metric("Performance", f"{performance_score:.1f}")
                    st.progress(min(performance_score/10, 1.0))
            
            with tab2:
                # Response & Wellness
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    composite_health = latest_data.get('COMPOSITE HEALTH SCORE', 0)
                    st.metric("Composite Health", f"{composite_health}")
                    
                with col2:
                    wellness_score = latest_data.get('MOST RECENT WELLNESS', 0)
                    st.metric("Wellness Score", f"{wellness_score}")
                    
                with col3:
                    composite_risk = latest_data.get('COMPOSITE RISK', 'GREEN')
                    st.metric("Composite Risk", composite_risk)
            
            with tab3:
                # Risk Flags - Z-Score Radar Chart
                risk_metrics = {
                    'Soreness': latest_data.get('SORENESS RISK  Z SCORE', 0),
                    'Hamstring': latest_data.get('HAMSTRING RISK Z SCORE', 0),
                    'Groin': latest_data.get('GROIN RISK Z SCORE', 0),
                    'Quad': latest_data.get('QUAD RISK Z SCORE', 0),
                    'Calf': latest_data.get('CALF RISK Z SCORE', 0),
                    'Low Back/SI': latest_data.get('LOW BACK/SI Z SCORE', 0)
                }
                
                # Create radar chart
                categories = list(risk_metrics.keys())
                values = list(risk_metrics.values())
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Risk Z-Scores',
                    line_color='red'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[-3, 3]
                        )),
                    showlegend=False,
                    title="Risk Flags (Z-Score Analysis)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                # Training Recommendation
                training_adj = latest_data.get('TRAINING ADJUSTMENT', 'NO ADJUSTMENT')
                
                if training_adj == 'NO ADJUSTMENT':
                    st.success(f"✅ {training_adj}")
                elif training_adj == 'MODIFIED TRAINING':
                    st.warning(f"⚠️ {training_adj}")
                elif training_adj == 'REDUCED TRAINING INTENSITY':
                    st.warning(f"⚠️ {training_adj}")
                elif training_adj == 'LIMITED':
                    st.error(f"🚨 {training_adj}")
                
                col1, col2 = st.columns(2)
                with col1:
                    workload_risk = latest_data.get('WORKLOAD W/ RISK MULTIPLIER', 0.0)
                    st.metric("Workload (w/ Risk)", f"{workload_risk:.2f}")
                with col2:
                    risk_mult = latest_data.get('RISK MULTIPLIER', 1.0)
                    st.metric("Risk Multiplier", f"{risk_mult}x")
            
            st.markdown("---")
            
            # Recent Trends (Last 7 Days)
            st.markdown("## 📈 Recent Trends (Last 7 Days)")
            
            # Get last 7 days of data
            last_7_days = athlete_df.tail(7)
            
            if len(last_7_days) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Workload Metrics Chart
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(
                        x=last_7_days['Date'],
                        y=last_7_days['VOLUME STEN'],
                        mode='lines+markers',
                        name='Volume',
                        line=dict(color='blue')
                    ))
                    fig1.add_trace(go.Scatter(
                        x=last_7_days['Date'],
                        y=last_7_days['INTENSITY STEN'],
                        mode='lines+markers',
                        name='Intensity',
                        line=dict(color='red')
                    ))
                    fig1.update_layout(
                        title="Workload Metrics",
                        xaxis_title="Date",
                        yaxis_title="Score",
                        height=350
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                with col2:
                    # Performance & Readiness Chart
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=last_7_days['Date'],
                        y=last_7_days['PERFORMANCE STEN'],
                        mode='lines+markers',
                        name='Performance',
                        line=dict(color='green')
                    ))
                    fig2.add_trace(go.Scatter(
                        x=last_7_days['Date'],
                        y=last_7_days['COMPOSITE HEALTH SCORE'],
                        mode='lines+markers',
                        name='Health Score',
                        line=dict(color='purple')
                    ))
                    fig2.update_layout(
                        title="Performance & Readiness",
                        xaxis_title="Date",
                        yaxis_title="Score",
                        height=350
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Risk Profile Chart
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(
                        x=last_7_days['Date'],
                        y=last_7_days['SORENESS RISK  Z SCORE'],
                        mode='lines+markers',
                        name='Soreness',
                        line=dict(color='orange')
                    ))
                    fig3.add_trace(go.Scatter(
                        x=last_7_days['Date'],
                        y=last_7_days['HAMSTRING RISK Z SCORE'],
                        mode='lines+markers',
                        name='Hamstring',
                        line=dict(color='red')
                    ))
                    fig3.update_layout(
                        title="Risk Profile (Z-Scores)",
                        xaxis_title="Date",
                        yaxis_title="Z-Score",
                        height=350
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    
                with col4:
                    # Distance & Speed Metrics
                    fig4 = go.Figure()
                    fig4.add_trace(go.Scatter(
                        x=last_7_days['Date'],
                        y=last_7_days['Distance'],
                        mode='lines+markers',
                        name='Distance',
                        line=dict(color='teal')
                    ))
                    if 'Distance at High Speed' in last_7_days.columns:
                        fig4.add_trace(go.Scatter(
                            x=last_7_days['Date'],
                            y=last_7_days['Distance at High Speed'],
                            mode='lines+markers',
                            name='High Speed Distance',
                            line=dict(color='navy')
                        ))
                    fig4.update_layout(
                        title="Distance & Speed Metrics",
                        xaxis_title="Date",
                        yaxis_title="Distance (m)",
                        height=350
                    )
                    st.plotly_chart(fig4, use_container_width=True)
            
            st.markdown("---")
            
            # Seasonal Performance Maxes
            st.markdown("## 🏆 Seasonal Performance Maxes")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                max_speed = athlete_df['Maximum Speed'].max() if 'Maximum Speed' in athlete_df.columns else 0
                st.metric("⚡ Maximum Speed", f"{max_speed:.2f} m/s")
                
            with col2:
                peak_accel = athlete_df['PEAK ACCEL'].max() if 'PEAK ACCEL' in athlete_df.columns else 0
                st.metric("🚀 Peak Acceleration", f"{peak_accel:.2f} m/s²")
                
            with col3:
                peak_decel = athlete_df['PEAK DECEL'].min() if 'PEAK DECEL' in athlete_df.columns else 0
                st.metric("✋ Peak Deceleration", f"{peak_decel:.2f} m/s²")
                
            with col4:
                max_jump = athlete_df['Maximum Jump Height'].max() if 'Maximum Jump Height' in athlete_df.columns else 0
                st.metric("⬆️ Maximum Jump", f"{max_jump:.2f} m")
            
            st.markdown("---")
            
            # Injury Impact Scoring
            st.markdown("## 🏥 Injury Impact Scoring")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Current Injury Status
                st.markdown("### Current Injury Status")
                
                injury_color = latest_data.get('INJURY IMPACT COLOR', 'GREEN')
                risk_mult = latest_data.get('RISK MULTIPLIER', 1.0)
                
                # Count affected areas (scores > 5)
                pain_areas = {
                    'Soreness': latest_data.get('SORENESS 0-10', 0),
                    'Hamstring': latest_data.get('HAMSTRING 0-10', 0),
                    'Groin': latest_data.get('GROIN 0-10', 0),
                    'Quad': latest_data.get('QUAD 0-10', 0),
                    'Calf': latest_data.get('CALF 0-10', 0),
                    'Low Back/SI': latest_data.get('LOW BACK 0-10', 0)
                }
                affected_areas = sum(1 for score in pain_areas.values() if score > 5)
                
                if injury_color == 'GREEN':
                    st.success(f"**Injury Impact:** {injury_color}")
                elif injury_color == 'YELLOW':
                    st.warning(f"**Injury Impact:** {injury_color}")
                elif injury_color == 'ORANGE':
                    st.warning(f"**Injury Impact:** {injury_color}")
                else:
                    st.error(f"**Injury Impact:** {injury_color}")
                
                st.metric("Risk Multiplier", f"{risk_mult}x")
                st.metric("Affected Areas", affected_areas)
                
            with col2:
                # Body Area Pain Scores
                st.markdown("### Body Area Pain Scores (0-10)")
                
                for area, score in pain_areas.items():
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.progress(score/10)
                    with col_b:
                        st.write(f"{area}: {score:.1f}")
            
            # 7-Day Injury Trend
            st.markdown("### 7-Day Pain Trend")
            
            if len(last_7_days) > 0:
                fig_injury = go.Figure()
                
                for area in ['SORENESS 0-10', 'HAMSTRING 0-10', 'GROIN 0-10', 'QUAD 0-10', 'CALF 0-10', 'LOW BACK 0-10']:
                    if area in last_7_days.columns:
                        fig_injury.add_trace(go.Scatter(
                            x=last_7_days['Date'],
                            y=last_7_days[area],
                            mode='lines+markers',
                            name=area.replace(' 0-10', '')
                        ))
                
                fig_injury.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Pain Score (0-10)",
                    height=400
                )
                
                st.plotly_chart(fig_injury, use_container_width=True)
            
            # Injury Risk Summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Find highest risk area
                highest_risk = max(pain_areas.items(), key=lambda x: x[1])
                st.info(f"⚠️ **Highest Risk Area:** {highest_risk[0]} ({highest_risk[1]:.1f})")
                
            with col2:
                # Check for increasing pain (simplified - would need historical data)
                st.info(f"📈 **Increasing Pain Areas:** Check trend above")
                
            with col3:
                # Find improving areas (scores < 4)
                improving = [area for area, score in pain_areas.items() if score < 4]
                if improving:
                    st.success(f"✅ **Improving Areas:** {', '.join(improving[:2])}")
                else:
                    st.info(f"✅ **Improving Areas:** None")
            
            st.markdown("---")
            
            # Training Load Heatmap (Last 30 Days)
            st.markdown("## 🔥 Training Load Heatmap (Last 30 Days)")
            
            # Get last 30 days of data
            last_30_days = athlete_df.tail(30)
            
            if len(last_30_days) > 0:
                # Create heatmap data
                heatmap_data = last_30_days[['Date', 'WORKLOAD STEN']].copy()
                heatmap_data['Day'] = heatmap_data['Date'].dt.day
                heatmap_data['Week'] = (heatmap_data.index // 7) + 1
                
                # Pivot for heatmap
                heatmap_pivot = heatmap_data.pivot_table(
                    values='WORKLOAD STEN', 
                    index='Week', 
                    columns='Day',
                    aggfunc='mean'
                )
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=heatmap_pivot.values,
                    x=[f"Day {i}" for i in heatmap_pivot.columns],
                    y=[f"Week {i}" for i in heatmap_pivot.index],
                    colorscale='RdYlGn_r',
                    colorbar=dict(title="Workload"),
                    text=np.round(heatmap_pivot.values, 1),
                    texttemplate="%{text}",
                    textfont={"size": 10}
                ))
                
                fig_heatmap.update_layout(
                    height=300,
                    xaxis_title="Day of Month",
                    yaxis_title="Week"
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown("---")
            
            # Recent Raw Metrics Table
            st.markdown("## 📋 Recent Raw Metrics")
            
            # Select columns for the table
            table_columns = ['Date', 'Total Load', 'Distance', 'Distance at High Speed', 
                            '# ACCELS', '# DECELS', 'Move Time', 'Impact Intensity']
            
            # Filter columns that exist
            available_columns = [col for col in table_columns if col in athlete_df.columns]
            
            # Show last 10 rows
            recent_metrics = athlete_df[available_columns].tail(10).sort_values('Date', ascending=False)
            
            # Format the dataframe for display
            recent_metrics_display = recent_metrics.copy()
            recent_metrics_display['Date'] = recent_metrics_display['Date'].dt.strftime('%Y-%m-%d')
            
            # Round numeric columns
            numeric_columns = recent_metrics_display.select_dtypes(include=[np.number]).columns
            recent_metrics_display[numeric_columns] = recent_metrics_display[numeric_columns].round(2)
            
            st.dataframe(recent_metrics_display, use_container_width=True, hide_index=True)
            
        else:
            st.warning("No data available for the selected athlete and date range.")

elif page == "Injury Impact":
    if df is None:
        st.error("⚠️ No data available. Please upload a CSV file using the sidebar.")
    else:
        # Filters at the top
        col1, col2, col3 = st.columns([2, 2, 8])
        
        with col1:
            # Athlete filter
            athletes = df['Person'].unique()
            selected_athlete = st.selectbox("Select Athlete", athletes, index=0)
        
        with col2:
            # Date range filter
            date_range = st.date_input(
                "Date Range",
                value=(df['Date'].min(), df['Date'].max()),
                min_value=df['Date'].min(),
                max_value=df['Date'].max()
            )
            
            # Handle both single date and date range
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = end_date = date_range[0] if isinstance(date_range, tuple) else date_range
        
        # Filter data
        athlete_df = df[
            (df['Person'] == selected_athlete) & 
            (df['Date'] >= pd.Timestamp(start_date)) & 
            (df['Date'] <= pd.Timestamp(end_date))
        ].sort_values('Date')
        
        if len(athlete_df) > 0:
            latest_data = athlete_df.iloc[-1]
            
            # Header
            st.markdown("# 🏥 Injury Impact Scoring")
            st.markdown("Comprehensive pain and injury risk assessment")
            
            st.markdown("---")
            
            # COMPOSITE RISK STATUS
            st.markdown("## 🎯 COMPOSITE RISK STATUS")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                risk_status = latest_data.get('COMPOSITE RISK', 'GREEN')
                if risk_status == 'GREEN':
                    st.success(f"**Status:** {risk_status}")
                    st.markdown("**READY TO TRAIN**")
                elif risk_status == 'YELLOW':
                    st.warning(f"**Status:** {risk_status}")
                    st.markdown("**MODIFIED TRAINING**")
                elif risk_status == 'ORANGE':
                    st.warning(f"**Status:** {risk_status}")
                    st.markdown("**REDUCED INTENSITY**")
                else:
                    st.error(f"**Status:** {risk_status}")
                    st.markdown("**LIMITED TRAINING**")
                    
            with col2:
                st.metric("Risk Multiplier", f"{latest_data.get('RISK MULTIPLIER', 1.0)}x")
                
            with col3:
                training_adj = latest_data.get('TRAINING ADJUSTMENT', 'NO ADJUSTMENT')
                st.metric("Training Adjustment", training_adj)
                
            with col4:
                workload_risk = latest_data.get('WORKLOAD W/ RISK MULTIPLIER', 0.0)
                st.metric("Workload w/ Risk", f"{workload_risk:.2f}")
            
            st.markdown("---")
            
            # Injury Impact Scoring Section
            st.markdown("## 📋 Injury Impact Scoring")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Current Injury Status
                st.markdown("### Current Injury Status")
                
                injury_color = latest_data.get('INJURY IMPACT COLOR', 'GREEN')
                risk_mult = latest_data.get('RISK MULTIPLIER', 1.0)
                
                # Count affected areas (scores > 5)
                pain_areas = {
                    'Soreness': latest_data.get('SORENESS 0-10', 0),
                    'Hamstring': latest_data.get('HAMSTRING 0-10', 0),
                    'Groin': latest_data.get('GROIN 0-10', 0),
                    'Quad': latest_data.get('QUAD 0-10', 0),
                    'Calf': latest_data.get('CALF 0-10', 0),
                    'Low Back/SI': latest_data.get('LOW BACK 0-10', 0)
                }
                affected_areas = sum(1 for score in pain_areas.values() if score > 5)
                
                if injury_color == 'GREEN':
                    st.success(f"**Impact Color:** {injury_color}")
                elif injury_color == 'YELLOW':
                    st.warning(f"**Impact Color:** {injury_color}")
                elif injury_color == 'ORANGE':
                    st.warning(f"**Impact Color:** {injury_color}")
                else:
                    st.error(f"**Impact Color:** {injury_color}")
                
                st.metric("Risk Multiplier", f"{risk_mult}x")
                st.metric("Affected Areas", affected_areas)
                
            with col2:
                # Body Area Pain Scores (0-10 Scale)
                st.markdown("### Body Area Pain Scores (0-10)")
                
                for area, score in pain_areas.items():
                    col_a, col_b, col_c = st.columns([2, 4, 1])
                    with col_a:
                        st.write(f"**{area}**")
                    with col_b:
                        # Color based on score
                        if score <= 3:
                            color = 'green'
                        elif score <= 6:
                            color = 'yellow'
                        elif score <= 8:
                            color = 'orange'
                        else:
                            color = 'red'
                        st.progress(score/10, text=f"{score:.1f}")
                    with col_c:
                        if score > 7:
                            st.write("⚠️")
            
            st.markdown("---")
            
            # 7-Day Pain Trend
            st.markdown("### 📈 7-Day Pain Trend")
            
            last_7_days = athlete_df.tail(7)
            
            if len(last_7_days) > 0:
                fig = go.Figure()
                
                pain_columns = {
                    'Soreness': 'SORENESS 0-10',
                    'Hamstring': 'HAMSTRING 0-10', 
                    'Groin': 'GROIN 0-10',
                    'Quad': 'QUAD 0-10',
                    'Calf': 'CALF 0-10',
                    'Low Back/SI': 'LOW BACK 0-10'
                }
                
                for label, col in pain_columns.items():
                    if col in last_7_days.columns:
                        fig.add_trace(go.Scatter(
                            x=last_7_days['Date'],
                            y=last_7_days[col],
                            mode='lines+markers',
                            name=label,
                            line=dict(width=2)
                        ))
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Pain Score (0-10)",
                    yaxis_range=[0, 10],
                    height=400,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Injury Risk Summary
            st.markdown("### 📊 Injury Risk Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Find highest risk area
                highest_risk = max(pain_areas.items(), key=lambda x: x[1])
                if highest_risk[1] > 0:
                    st.error(f"⚠️ **Highest Risk Area**")
                    st.write(f"{highest_risk[0]}: {highest_risk[1]:.1f}/10")
                else:
                    st.success(f"✅ **No Risk Areas**")
                
            with col2:
                # Check for increasing pain areas
                if len(last_7_days) >= 2:
                    recent = last_7_days.iloc[-1]
                    prev = last_7_days.iloc[-2]
                    increasing = []
                    for area, col in pain_columns.items():
                        if col in recent.index and col in prev.index:
                            if recent[col] > prev[col] + 0.5:
                                increasing.append(area)
                    
                    if increasing:
                        st.warning(f"📈 **Increasing Pain Areas**")
                        st.write(", ".join(increasing))
                    else:
                        st.info(f"➡️ **Stable Pain Levels**")
                else:
                    st.info(f"📊 **Need More Data**")
                
            with col3:
                # Find improving areas (scores < 4)
                improving = [area for area, score in pain_areas.items() if score < 4]
                if improving:
                    st.success(f"✅ **Low Risk Areas**")
                    st.write(", ".join(improving))
                else:
                    st.warning(f"⚠️ **No Low Risk Areas**")
            
            st.markdown("---")
            
            # 30-Day Pain History
            st.markdown("## 📅 30-Day Pain History")
            
            last_30_days = athlete_df.tail(30)
            
            if len(last_30_days) > 0:
                fig = go.Figure()
                
                # Create stacked area chart for pain history
                for label, col in pain_columns.items():
                    if col in last_30_days.columns:
                        fig.add_trace(go.Scatter(
                            x=last_30_days['Date'],
                            y=last_30_days[col],
                            mode='lines',
                            name=label,
                            stackgroup='one',
                            line=dict(width=0.5)
                        ))
                
                fig.update_layout(
                    title="30-Day Pain History (Stacked)",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Pain Score",
                    height=500,
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional heatmap view
                st.markdown("### 🔥 Pain Intensity Heatmap")
                
                # Prepare data for heatmap
                heatmap_data = []
                dates = []
                
                for _, row in last_30_days.iterrows():
                    row_data = []
                    for col in pain_columns.values():
                        if col in row.index:
                            row_data.append(row[col])
                        else:
                            row_data.append(0)
                    heatmap_data.append(row_data)
                    dates.append(row['Date'].strftime('%m/%d'))
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=np.array(heatmap_data).T,
                    x=dates,
                    y=list(pain_columns.keys()),
                    colorscale='RdYlGn_r',
                    colorbar=dict(title="Pain Score"),
                    text=np.round(np.array(heatmap_data).T, 1),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    zmid=5,
                    zmin=0,
                    zmax=10
                ))
                
                fig_heatmap.update_layout(
                    title="Body Area Pain Intensity Over Time",
                    xaxis_title="Date",
                    yaxis_title="Body Area",
                    height=400
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.warning("No data available for the last 30 days")
        else:
            st.warning("No data available for the selected athlete and date range.")

else:
    st.info(f"The '{page}' page is under development. Please select 'Athlete Profile' or 'Injury Impact' from the sidebar.")