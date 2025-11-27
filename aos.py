import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Athlete Operating System - AOS",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CHART LAYOUT DEFAULTS (White Background Standard)
# ============================================================================

CHART_FONT = dict(
    family="Inter, -apple-system, sans-serif",
    size=12,
    color="#374151"  # Dark gray for white bg
)

CHART_LAYOUT = dict(
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=CHART_FONT,
    margin=dict(l=50, r=30, t=40, b=50),
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='#E5E7EB',
        linecolor='#D1D5DB',
        linewidth=1,
        tickfont=dict(size=11, color='#4B5563'),
        title_font=dict(size=12, color='#374151')
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='#E5E7EB',
        linecolor='#D1D5DB',
        linewidth=1,
        tickfont=dict(size=11, color='#4B5563'),
        title_font=dict(size=12, color='#374151')
    ),
    legend=dict(
        font=dict(size=11, color='#374151'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#E5E7EB',
        borderwidth=1
    )
)

def apply_chart_style(fig, height=300, show_legend=True):
    """Apply standard white-background styling to any plotly figure"""
    fig.update_layout(
        height=height,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=CHART_FONT,
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color='#374151'),
            bgcolor='rgba(255,255,255,0.9)'
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#E5E7EB',
        linecolor='#D1D5DB',
        linewidth=1,
        tickfont=dict(size=11, color='#4B5563'),
        title_font=dict(size=12, color='#374151')
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#E5E7EB',
        linecolor='#D1D5DB',
        linewidth=1,
        tickfont=dict(size=11, color='#4B5563'),
        title_font=dict(size=12, color='#374151')
    )
    return fig

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main { background-color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #F9FAFB; }
    [data-testid="stSidebar"] * { color: #1F2937 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #111827 !important; }
    
    .sten-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 12px; color: white; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 5px 0;
    }
    .sten-score { font-size: 36px; font-weight: bold; margin: 5px 0; }
    .sten-label { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9; }
    
    .risk-card {
        padding: 15px; border-radius: 10px; text-align: center; margin: 5px 0;
    }
    .risk-green { background-color: #10B981; color: white; }
    .risk-yellow { background-color: #F59E0B; color: white; }
    .risk-red { background-color: #EF4444; color: white; }
    
    .metric-card {
        background-color: #F3F4F6; padding: 15px; border-radius: 10px;
        text-align: center; border: 1px solid #E5E7EB;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #3B82F6; }
    .metric-label { font-size: 11px; color: #6B7280; text-transform: uppercase; }
    
    h1, h2, h3 { color: #111827; }
    p, span, div { color: #374151; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Data labels styling */
    .plotly .text { font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def convert_sheet_url_to_csv(sheet_url):
    """Convert Google Sheets URL to CSV export URL"""
    if '/edit' in sheet_url:
        sheet_id = sheet_url.split('/d/')[1].split('/')[0]
        if '#gid=' in sheet_url:
            gid = sheet_url.split('#gid=')[1].split('&')[0]
        elif 'gid=' in sheet_url:
            gid = sheet_url.split('gid=')[1].split('&')[0]
        else:
            gid = '0'
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    return None

@st.cache_data(ttl=60)  # Cache for 1 minute
def load_data_from_url(sheet_url):
    """Load data from Google Sheets URL"""
    try:
        csv_url = convert_sheet_url_to_csv(sheet_url)
        if csv_url is None:
            return None
        df = pd.read_csv(csv_url)
        
        # Convert date columns
        date_cols = ['Date', 'DATE', 'Day']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Standardize name column
        name_cols = ['Person', 'NAME', 'Name', 'Athlete', 'Athlete Name']
        for col in name_cols:
            if col in df.columns:
                df = df.rename(columns={col: 'Athlete'})
                break
        return df
    except Exception as e:
        st.error(f"Error loading sheet: {e}")
        return None

# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================

def calculate_z_score_per_athlete(df, col):
    """Calculate Z-score per athlete (compared to their own baseline)"""
    def z_score(x):
        mean = x.mean()
        std = x.std()
        if std == 0 or pd.isna(std) or len(x) < 2:
            return pd.Series([0] * len(x), index=x.index)
        return (x - mean) / std
    
    return df.groupby('Athlete')[col].transform(z_score)

def calculate_sten(z_score):
    """Convert Z-score to STEN (1-10 scale)
    STEN = 5.5 + (2 * Z)
    Capped at 1-10
    """
    sten = 5.5 + (2 * z_score)
    return np.clip(sten, 1, 10)

def calculate_acwr(df, col, athlete_col='Athlete'):
    """Calculate Acute:Chronic Workload Ratio per athlete
    Acute = 7-day rolling mean
    Chronic = 28-day rolling mean
    """
    def acwr_calc(group):
        group = group.sort_values('Date')
        acute = group[col].rolling(window=7, min_periods=1).mean()
        chronic = group[col].rolling(window=28, min_periods=7).mean()
        acwr = acute / chronic
        acwr = acwr.replace([np.inf, -np.inf], np.nan)
        return acwr
    
    if 'Date' not in df.columns:
        return pd.Series([1.0] * len(df), index=df.index)
    
    return df.groupby(athlete_col, group_keys=False).apply(
        lambda g: pd.Series(acwr_calc(g).values, index=g.index)
    )

def normalize_per_athlete(df, col):
    """Normalize column to 0-10 scale per athlete"""
    def norm(x):
        min_val, max_val = x.min(), x.max()
        if max_val == min_val or pd.isna(max_val) or pd.isna(min_val):
            return pd.Series([5.0] * len(x), index=x.index)
        return ((x - min_val) / (max_val - min_val)) * 10
    
    return df.groupby('Athlete')[col].transform(norm)

def calculate_all_metrics(df):
    """Calculate Z-scores, STEN scores, A:C ratios, and risk metrics PER ATHLETE"""
    df = df.copy()
    
    # Ensure we have Athlete column
    if 'Athlete' not in df.columns:
        df['Athlete'] = 'Unknown'
    
    # Sort by athlete and date for proper rolling calculations
    if 'Date' in df.columns:
        df = df.sort_values(['Athlete', 'Date'])
    
    # ===== VOLUME METRICS =====
    volume_cols = ['Distance', 'Total Load', 'Total Impact', 'Total Horizontal Impact', 'Total Vertical Impact']
    volume_cols_present = [c for c in volume_cols if c in df.columns]
    
    if volume_cols_present:
        # Calculate Z-score for each volume metric per athlete
        volume_z_list = []
        for col in volume_cols_present:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            z_col = f'{col}_z'
            df[z_col] = calculate_z_score_per_athlete(df, col)
            volume_z_list.append(z_col)
        
        # Average of all volume Z-scores
        df['CALC_VOLUME_Z'] = df[volume_z_list].mean(axis=1)
    else:
        df['CALC_VOLUME_Z'] = 0.0
    
    # ===== INTENSITY METRICS =====
    intensity_cols = ['# ACCELS', '# DECELS', 'PEAK ACCEL', 'PEAK DECEL', 'Distance at High Speed', 'Distance at Very High Speed', 'Impact Intensity']
    intensity_cols_present = [c for c in intensity_cols if c in df.columns]
    
    if intensity_cols_present:
        intensity_z_list = []
        for col in intensity_cols_present:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            z_col = f'{col}_int_z'
            df[z_col] = calculate_z_score_per_athlete(df, col)
            intensity_z_list.append(z_col)
        
        df['CALC_INTENSITY_Z'] = df[intensity_z_list].mean(axis=1)
    else:
        df['CALC_INTENSITY_Z'] = 0.0
    
    # ===== ASYMMETRY METRICS =====
    asym_cols = [c for c in df.columns if 'Asymmetry' in c]
    
    if asym_cols:
        asym_z_list = []
        for col in asym_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            # Use absolute value for asymmetry (magnitude matters, not direction)
            df[f'{col}_abs'] = df[col].abs()
            z_col = f'{col}_asym_z'
            df[z_col] = calculate_z_score_per_athlete(df, f'{col}_abs')
            asym_z_list.append(z_col)
        
        df['CALC_ASYMMETRY_Z'] = df[asym_z_list].mean(axis=1)
    else:
        df['CALC_ASYMMETRY_Z'] = 0.0
    
    # ===== PERFORMANCE METRICS =====
    perf_cols = ['Maximum Speed', 'PEAK ACCEL', 'PEAK DECEL', 'Push-off Intensity']
    perf_cols_present = [c for c in perf_cols if c in df.columns]
    
    if perf_cols_present:
        perf_z_list = []
        for col in perf_cols_present:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            z_col = f'{col}_perf_z'
            df[z_col] = calculate_z_score_per_athlete(df, col)
            perf_z_list.append(z_col)
        
        df['CALC_PERFORMANCE_Z'] = df[perf_z_list].mean(axis=1)
    else:
        df['CALC_PERFORMANCE_Z'] = 0.0
    
    # ===== STEN SCORES (1-10 scale) =====
    df['CALC_VOLUME_STEN'] = calculate_sten(df['CALC_VOLUME_Z'])
    df['CALC_INTENSITY_STEN'] = calculate_sten(df['CALC_INTENSITY_Z'])
    df['CALC_SYMMETRY_STEN'] = calculate_sten(-df['CALC_ASYMMETRY_Z'])  # Inverse: lower asymmetry = higher STEN
    df['CALC_PERFORMANCE_STEN'] = calculate_sten(df['CALC_PERFORMANCE_Z'])
    
    # ===== WORKLOAD STEN =====
    df['CALC_WORKLOAD_STEN'] = (df['CALC_VOLUME_STEN'] + df['CALC_INTENSITY_STEN']) / 2
    
    # ===== ACUTE:CHRONIC WORKLOAD RATIO (A:C) =====
    if 'Total Load' in df.columns and 'Date' in df.columns:
        df['Total Load'] = pd.to_numeric(df['Total Load'], errors='coerce').fillna(0)
        df['CALC_ACWR_LOAD'] = calculate_acwr(df, 'Total Load')
        
        # Also calculate for Volume STEN
        df['CALC_ACWR_VOLUME'] = calculate_acwr(df, 'CALC_VOLUME_STEN')
        
        # Acute (7-day) and Chronic (28-day) values for display
        df['CALC_ACUTE_LOAD'] = df.groupby('Athlete')['Total Load'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        df['CALC_CHRONIC_LOAD'] = df.groupby('Athlete')['Total Load'].transform(
            lambda x: x.rolling(window=28, min_periods=7).mean()
        )
    else:
        df['CALC_ACWR_LOAD'] = 1.0
        df['CALC_ACWR_VOLUME'] = 1.0
        df['CALC_ACUTE_LOAD'] = 0.0
        df['CALC_CHRONIC_LOAD'] = 0.0
    
    # ===== NORMALIZED INPUTS FOR RISK (0-10 scale, per athlete) =====
    
    # HSR Stress
    if 'Distance at High Speed' in df.columns:
        df['Distance at High Speed'] = pd.to_numeric(df['Distance at High Speed'], errors='coerce').fillna(0)
        df['HSR_Stress'] = normalize_per_athlete(df, 'Distance at High Speed')
    else:
        df['HSR_Stress'] = 5.0
    
    # Max Speed
    if 'Maximum Speed' in df.columns:
        df['Maximum Speed'] = pd.to_numeric(df['Maximum Speed'], errors='coerce').fillna(0)
        df['MaxSpeed_Norm'] = normalize_per_athlete(df, 'Maximum Speed')
    else:
        df['MaxSpeed_Norm'] = 5.0
    
    # Decel Stress (use absolute value)
    if 'PEAK DECEL' in df.columns:
        df['PEAK DECEL'] = pd.to_numeric(df['PEAK DECEL'], errors='coerce').fillna(0)
        df['PEAK_DECEL_ABS'] = df['PEAK DECEL'].abs()
        df['Decel_Stress'] = normalize_per_athlete(df, 'PEAK_DECEL_ABS')
    elif '# DECELS' in df.columns:
        df['# DECELS'] = pd.to_numeric(df['# DECELS'], errors='coerce').fillna(0)
        df['Decel_Stress'] = normalize_per_athlete(df, '# DECELS')
    else:
        df['Decel_Stress'] = 5.0
    
    # Accel Stress
    if 'PEAK ACCEL' in df.columns:
        df['PEAK ACCEL'] = pd.to_numeric(df['PEAK ACCEL'], errors='coerce').fillna(0)
        df['Accel_Stress'] = normalize_per_athlete(df, 'PEAK ACCEL')
    elif '# ACCELS' in df.columns:
        df['# ACCELS'] = pd.to_numeric(df['# ACCELS'], errors='coerce').fillna(0)
        df['Accel_Stress'] = normalize_per_athlete(df, '# ACCELS')
    else:
        df['Accel_Stress'] = 5.0
    
    # Asymmetry Score (average of all asymmetry columns, absolute)
    if asym_cols:
        abs_asym_cols = [f'{c}_abs' for c in asym_cols if f'{c}_abs' in df.columns]
        if abs_asym_cols:
            df['Avg_Asymmetry'] = df[abs_asym_cols].mean(axis=1)
            df['Asymmetry_Norm'] = normalize_per_athlete(df, 'Avg_Asymmetry')
        else:
            df['Asymmetry_Norm'] = 5.0
    else:
        df['Asymmetry_Norm'] = 5.0
    
    # Impact Load
    if 'Total Impact' in df.columns:
        df['Total Impact'] = pd.to_numeric(df['Total Impact'], errors='coerce').fillna(0)
        df['ImpactLoad_Norm'] = normalize_per_athlete(df, 'Total Impact')
    elif 'Impact Intensity' in df.columns:
        df['Impact Intensity'] = pd.to_numeric(df['Impact Intensity'], errors='coerce').fillna(0)
        df['ImpactLoad_Norm'] = normalize_per_athlete(df, 'Impact Intensity')
    else:
        df['ImpactLoad_Norm'] = 5.0
    
    # Vertical Impact
    if 'Total Vertical Impact' in df.columns:
        df['Total Vertical Impact'] = pd.to_numeric(df['Total Vertical Impact'], errors='coerce').fillna(0)
        df['VerticalImpact_Norm'] = normalize_per_athlete(df, 'Total Vertical Impact')
    else:
        df['VerticalImpact_Norm'] = 5.0
    
    # Volume Score (normalized load)
    if 'Total Load' in df.columns:
        df['Volume_Norm'] = normalize_per_athlete(df, 'Total Load')
    elif 'Distance' in df.columns:
        df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
        df['Volume_Norm'] = normalize_per_athlete(df, 'Distance')
    else:
        df['Volume_Norm'] = 5.0
    
    # ===== SITE-SPECIFIC RISK CALCULATIONS =====
    
    # Hamstring Risk: 0.35×HSR + 0.25×MaxSpeed + 0.20×Decel + 0.20×Asymmetry
    df['CALC_HAMSTRING_RISK'] = (
        0.35 * df['HSR_Stress'] +
        0.25 * df['MaxSpeed_Norm'] +
        0.20 * df['Decel_Stress'] +
        0.20 * df['Asymmetry_Norm']
    )
    
    # Groin Risk: 0.40×Decel + 0.25×Accel + 0.20×Asymmetry + 0.15×HSR
    df['CALC_GROIN_RISK'] = (
        0.40 * df['Decel_Stress'] +
        0.25 * df['Accel_Stress'] +
        0.20 * df['Asymmetry_Norm'] +
        0.15 * df['HSR_Stress']
    )
    
    # Hip Flexor Risk: 0.50×Accel + 0.20×HSR + 0.15×Volume + 0.15×Asymmetry
    df['CALC_HIPFLEXOR_RISK'] = (
        0.50 * df['Accel_Stress'] +
        0.20 * df['HSR_Stress'] +
        0.15 * df['Volume_Norm'] +
        0.15 * df['Asymmetry_Norm']
    )
    
    # Quad Risk: 0.45×Impact + 0.30×Decel + 0.15×Volume + 0.10×Asymmetry
    df['CALC_QUAD_RISK'] = (
        0.45 * df['ImpactLoad_Norm'] +
        0.30 * df['Decel_Stress'] +
        0.15 * df['Volume_Norm'] +
        0.10 * df['Asymmetry_Norm']
    )
    
    # Calf Risk: 0.40×VerticalImpact + 0.30×HSR + 0.20×Volume + 0.10×Asymmetry
    df['CALC_CALF_RISK'] = (
        0.40 * df['VerticalImpact_Norm'] +
        0.30 * df['HSR_Stress'] +
        0.20 * df['Volume_Norm'] +
        0.10 * df['Asymmetry_Norm']
    )
    
    # Low Back Risk: 0.40×Asymmetry + 0.35×Volume + 0.25×Decel
    df['CALC_LOWBACK_RISK'] = (
        0.40 * df['Asymmetry_Norm'] +
        0.35 * df['Volume_Norm'] +
        0.25 * df['Decel_Stress']
    )
    
    # ===== COMPOSITE SCORES =====
    
    # Average Soreness Risk
    df['CALC_AVG_RISK'] = df[['CALC_HAMSTRING_RISK', 'CALC_GROIN_RISK', 'CALC_HIPFLEXOR_RISK',
                              'CALC_QUAD_RISK', 'CALC_CALF_RISK', 'CALC_LOWBACK_RISK']].mean(axis=1)
    
    # Risk-adjusted workload: Workload × (1 + AvgRisk/20)
    df['CALC_WORKLOAD_RISK_ADJ'] = df['CALC_WORKLOAD_STEN'] * (1 + df['CALC_AVG_RISK'] / 20)
    
    return df

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🏋️ AOS - Platform")
st.sidebar.markdown("---")

# File uploaders for each sheet
st.sidebar.subheader("📁 Google Sheet URLs")
physical_url = st.sidebar.text_input("Physical Sheet URL", key='physical')
wellness_url = st.sidebar.text_input("Wellness Sheet URL", key='wellness')
injury_url = st.sidebar.text_input("Injury Sheet URL", key='injury')
info_url = st.sidebar.text_input("Info Sheet URL", key='info')

st.sidebar.markdown("---")

# Data Refresh Controls
st.sidebar.subheader("🔄 Data Refresh")

# Manual refresh button
if st.sidebar.button("🔄 Refresh Data Now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
if auto_refresh:
    refresh_interval = st.sidebar.selectbox(
        "Refresh interval",
        options=[1, 5, 15, 30, 60],
        index=2,  # Default 15 min
        format_func=lambda x: f"{x} min"
    )
    st.sidebar.caption(f"⏱️ Auto-refresh every {refresh_interval} min")
    
    # Meta refresh tag for auto-reload
    st.markdown(
        f'<meta http-equiv="refresh" content="{refresh_interval * 60}">',
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Team", "🏃 Physical", "❤️ Wellness", "🩹 Injury", "ℹ️ Info"]
)

# Load data if URLs provided
data = {}
if physical_url:
    df_physical = load_data_from_url(physical_url)
    if df_physical is not None:
        data['physical'] = calculate_all_metrics(df_physical)
if wellness_url:
    df_wellness = load_data_from_url(wellness_url)
    if df_wellness is not None:
        data['wellness'] = df_wellness
if injury_url:
    df_injury = load_data_from_url(injury_url)
    if df_injury is not None:
        data['injury'] = df_injury
if info_url:
    df_info = load_data_from_url(info_url)
    if df_info is not None:
        data['info'] = df_info

# Athlete filter
if 'physical' in data and 'Athlete' in data['physical'].columns:
    athletes = ['All Athletes'] + sorted(data['physical']['Athlete'].dropna().unique().tolist())
    selected_athlete = st.sidebar.selectbox("Select Athlete", athletes)
else:
    selected_athlete = "All Athletes"

# Date filter - combine from all sheets with Date column
all_dates = []
for sheet_name in ['physical', 'wellness', 'injury', 'info']:
    if sheet_name in data and 'Date' in data[sheet_name].columns:
        all_dates.extend(data[sheet_name]['Date'].dropna().tolist())

if all_dates:
    min_date = pd.Timestamp(min(all_dates))
    max_date = pd.Timestamp(max(all_dates))
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
    else:
        date_range = None
else:
    date_range = None

# ============================================================================
# PAGE 1: TEAM OVERVIEW
# ============================================================================

if page == "🏠 Team":
    st.title("Team Overview Dashboard")
    
    if 'physical' not in data:
        st.warning("📋 Paste Physical sheet URL in sidebar to view team data")
    else:
        phys = data['physical'].copy()
        well = data.get('wellness', pd.DataFrame()).copy()
        info = data.get('info', pd.DataFrame()).copy()
        injury = data.get('injury', pd.DataFrame()).copy()
        
        # Standardize name columns
        if 'Name' in well.columns:
            well = well.rename(columns={'Name': 'Athlete'})
        if 'NAME' in info.columns:
            info = info.rename(columns={'NAME': 'Athlete'})
        if 'Name' in injury.columns:
            injury = injury.rename(columns={'Name': 'Athlete'})
        
        # Filter by date
        if date_range and len(date_range) == 2 and 'Date' in phys.columns:
            phys = phys[(phys['Date'].dt.date >= date_range[0]) & (phys['Date'].dt.date <= date_range[1])]
        
        # Get athletes from ALL sheets (union)
        all_athletes = set()
        if 'Athlete' in phys.columns:
            all_athletes.update(phys['Athlete'].dropna().unique())
        if 'Athlete' in well.columns:
            all_athletes.update(well['Athlete'].dropna().unique())
        if 'Athlete' in info.columns:
            all_athletes.update(info['Athlete'].dropna().unique())
        if 'Athlete' in injury.columns:
            all_athletes.update(injury['Athlete'].dropna().unique())
        athletes = sorted(list(all_athletes))
        
        # ===== ROW 1: KEY METRICS =====
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Athletes</div>
                <div class="metric-value">{len(athletes)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_sessions = len(phys)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Sessions</div>
                <div class="metric-value">{total_sessions}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_time = phys['Move Time'].sum() / 60 if 'Move Time' in phys.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Training (hrs)</div>
                <div class="metric-value">{total_time:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_dist = phys['Distance'].sum() / 1000 if 'Distance' in phys.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Distance (km)</div>
                <div class="metric-value">{total_dist:,.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            avg_wellness = well['Wellness %'].mean() if 'Wellness %' in well.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Wellness %</div>
                <div class="metric-value">{avg_wellness:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ===== ROW 2: ATHLETE STATUS TRAFFIC LIGHT =====
        st.subheader("🚦 Athlete Status Overview")
        
        status_data = []
        for athlete in athletes:
            ath_phys = phys[phys['Athlete'] == athlete]
            ath_well = well[well['Athlete'] == athlete] if len(well) > 0 and 'Athlete' in well.columns else pd.DataFrame()
            ath_info = info[info['Athlete'] == athlete] if len(info) > 0 and 'Athlete' in info.columns else pd.DataFrame()
            ath_injury = injury[injury['Athlete'] == athlete] if len(injury) > 0 and 'Athlete' in injury.columns else pd.DataFrame()
            
            # Get latest physical values
            if len(ath_phys) > 0 and 'Date' in ath_phys.columns:
                latest_phys = ath_phys.sort_values('Date').iloc[-1]
            elif len(ath_phys) > 0:
                latest_phys = ath_phys.iloc[-1]
            else:
                latest_phys = pd.Series()
            
            # Get latest wellness values
            if len(ath_well) > 0 and 'Date' in ath_well.columns:
                latest_well = ath_well.sort_values('Date').iloc[-1]
            elif len(ath_well) > 0:
                latest_well = ath_well.iloc[-1]
            else:
                latest_well = pd.Series()
            
            # Calculate workload - use calculated STEN or fallback to raw data
            if 'CALC_WORKLOAD_STEN' in latest_phys.index:
                workload = latest_phys['CALC_WORKLOAD_STEN']
            elif 'Total Load' in latest_phys.index and len(ath_phys) > 0:
                # Normalize to 0-10 scale based on athlete's data
                load_val = latest_phys['Total Load']
                load_min = ath_phys['Total Load'].min()
                load_max = ath_phys['Total Load'].max()
                if load_max > load_min:
                    workload = ((load_val - load_min) / (load_max - load_min)) * 10
                else:
                    workload = 5
            else:
                workload = 5
            
            # Get wellness and readiness
            if 'Wellness %' in latest_well.index:
                wellness = latest_well['Wellness %']
            else:
                wellness = 80
            
            if 'READINESS TO TRAIN' in latest_well.index:
                readiness = latest_well['READINESS TO TRAIN']
            else:
                readiness = 7
            
            # Ensure numeric values
            workload = float(workload) if pd.notna(workload) else 5
            wellness = float(wellness) if pd.notna(wellness) else 80
            readiness = float(readiness) if pd.notna(readiness) else 7
            
            # Get injury impact
            injury_score = ath_injury['FINAL INJURY IMPACT SCORE'].iloc[0] if len(ath_injury) > 0 and 'FINAL INJURY IMPACT SCORE' in ath_injury.columns else 0
            injury_score = float(injury_score) if pd.notna(injury_score) else 0
            
            # Get risk tier from info
            risk_tier = ath_info['RISK TIER'].iloc[0] if len(ath_info) > 0 and 'RISK TIER' in ath_info.columns else 'Green'
            
            # Calculate composite status
            wellness_score = wellness / 10 if wellness > 10 else wellness
            readiness_score = readiness
            injury_penalty = min(injury_score / 10, 3) if injury_score > 0 else 0
            
            composite = (wellness_score * 0.3 + readiness_score * 0.4 + (10 - workload) * 0.3) - injury_penalty
            
            if composite >= 7:
                status = 'GREEN'
                status_color = '#10B981'
            elif composite >= 5:
                status = 'YELLOW'
                status_color = '#F59E0B'
            else:
                status = 'RED'
                status_color = '#EF4444'
            
            status_data.append({
                'Athlete': athlete,
                'Workload': workload,
                'Wellness': wellness,
                'Readiness': readiness,
                'Injury Impact': injury_score,
                'Composite': composite,
                'Status': status,
                'Color': status_color
            })
        
        if status_data:
            # Display as cards
            cols = st.columns(min(len(status_data), 5))
            for idx, ath in enumerate(status_data):
                col_idx = idx % min(len(status_data), 5)
                with cols[col_idx]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {ath['Color']}22, {ath['Color']}11); 
                                border: 2px solid {ath['Color']}; border-radius: 12px; padding: 15px; 
                                text-align: center; margin: 5px 0;">
                        <div style="font-size: 16px; font-weight: bold; color: #111827;">{ath['Athlete']}</div>
                        <div style="font-size: 28px; font-weight: bold; color: {ath['Color']}; margin: 5px 0;">
                            {ath['Status']}
                        </div>
                        <div style="font-size: 11px; color: #6B7280;">
                            W:{ath['Wellness']:.0f}% | R:{ath['Readiness']:.0f} | L:{ath['Workload']:.1f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ===== ROW 3: READINESS VS WORKLOAD QUADRANT =====
        st.subheader("📊 Readiness vs Workload Matrix")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if status_data:
                quad_df = pd.DataFrame(status_data)
                
                fig = go.Figure()
                
                # Add quadrant backgrounds
                fig.add_shape(type="rect", x0=0, y0=5, x1=5, y1=10, fillcolor="rgba(16, 185, 129, 0.1)", line_width=0)  # Low load, High readiness - GOOD
                fig.add_shape(type="rect", x0=5, y0=5, x1=10, y1=10, fillcolor="rgba(245, 158, 11, 0.1)", line_width=0)  # High load, High readiness - MONITOR
                fig.add_shape(type="rect", x0=0, y0=0, x1=5, y1=5, fillcolor="rgba(59, 130, 246, 0.1)", line_width=0)  # Low load, Low readiness - RECOVER
                fig.add_shape(type="rect", x0=5, y0=0, x1=10, y1=5, fillcolor="rgba(239, 68, 68, 0.1)", line_width=0)  # High load, Low readiness - DANGER
                
                # Add athletes
                fig.add_trace(go.Scatter(
                    x=quad_df['Workload'],
                    y=quad_df['Readiness'],
                    mode='markers+text',
                    marker=dict(size=20, color=quad_df['Color'].tolist(), line=dict(width=2, color='white')),
                    text=quad_df['Athlete'],
                    textposition='top center',
                    textfont=dict(color='#374151', size=10),
                    hovertemplate='<b>%{text}</b><br>Workload: %{x:.1f}<br>Readiness: %{y:.1f}<extra></extra>'
                ))
                
                # Add quadrant labels
                fig.add_annotation(x=2.5, y=7.5, text="✅ OPTIMAL", showarrow=False, font=dict(color='#10B981', size=12))
                fig.add_annotation(x=7.5, y=7.5, text="⚡ PUSHING", showarrow=False, font=dict(color='#F59E0B', size=12))
                fig.add_annotation(x=2.5, y=2.5, text="🔄 RECOVERING", showarrow=False, font=dict(color='#3B82F6', size=12))
                fig.add_annotation(x=7.5, y=2.5, text="🚨 DANGER", showarrow=False, font=dict(color='#EF4444', size=12))
                
                fig.update_layout(
                    height=400,
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='#374151', size=12),
                    xaxis=dict(range=[0, 10], title='Workload STEN', gridcolor='#E5E7EB', tickfont=dict(size=11, color='#4B5563')),
                    yaxis=dict(range=[0, 10], title='Readiness', gridcolor='#E5E7EB', tickfont=dict(size=11, color='#4B5563')),
                    margin=dict(l=50, r=40, t=40, b=50)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Quadrant Guide:**")
            st.markdown("""
            <div style="background-color: #F3F4F6; padding: 15px; border-radius: 10px; border: 1px solid #E5E7EB;">
                <div style="margin-bottom: 10px;">
                    <span style="color: #10B981; font-weight: bold;">✅ OPTIMAL</span><br>
                    <span style="font-size: 11px; color: #6B7280;">High readiness, manageable load</span>
                </div>
                <div style="margin-bottom: 10px;">
                    <span style="color: #F59E0B; font-weight: bold;">⚡ PUSHING</span><br>
                    <span style="font-size: 11px; color: #6B7280;">High readiness, high load - monitor</span>
                </div>
                <div style="margin-bottom: 10px;">
                    <span style="color: #3B82F6; font-weight: bold;">🔄 RECOVERING</span><br>
                    <span style="font-size: 11px; color: #6B7280;">Low readiness, low load - resting</span>
                </div>
                <div>
                    <span style="color: #EF4444; font-weight: bold;">🚨 DANGER</span><br>
                    <span style="font-size: 11px; color: #6B7280;">Low readiness, high load - risk!</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ===== ROW 4: LEADERBOARDS =====
        st.subheader("🏆 Training Leaderboards")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Training Time (hrs)**")
            if 'Move Time' in phys.columns:
                time_leader = phys.groupby('Athlete')['Move Time'].sum().sort_values(ascending=False) / 60
                
                fig = go.Figure(go.Bar(
                    x=time_leader.values,
                    y=time_leader.index,
                    orientation='h',
                    marker_color='#3B82F6',
                    text=[f"{v:.1f}" for v in time_leader.values],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    height=250,
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='#374151', size=12),
                    margin=dict(l=80, r=40, t=10, b=40),
                    xaxis=dict(title='Hours')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Distance (km)**")
            if 'Distance' in phys.columns:
                dist_leader = phys.groupby('Athlete')['Distance'].sum().sort_values(ascending=False) / 1000
                
                fig = go.Figure(go.Bar(
                    x=dist_leader.values,
                    y=dist_leader.index,
                    orientation='h',
                    marker_color='#10B981',
                    text=[f"{v:.1f}" for v in dist_leader.values],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    height=250,
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='#374151', size=12),
                    margin=dict(l=80, r=40, t=10, b=40),
                    xaxis=dict(title='km')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("**Total Load**")
            if 'Total Load' in phys.columns:
                load_leader = phys.groupby('Athlete')['Total Load'].sum().sort_values(ascending=False) / 1000
                
                fig = go.Figure(go.Bar(
                    x=load_leader.values,
                    y=load_leader.index,
                    orientation='h',
                    marker_color='#F59E0B',
                    text=[f"{v:.0f}k" for v in load_leader.values],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    height=250,
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='#374151', size=12),
                    margin=dict(l=80, r=40, t=10, b=40),
                    xaxis=dict(title='Load (k)')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # ===== ROW 5: TEAM RISK RADAR =====
        st.subheader("⚠️ Team Average Risk Profile")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Calculate team average risks
            risk_cols = ['CALC_HAMSTRING_RISK', 'CALC_GROIN_RISK', 'CALC_HIPFLEXOR_RISK', 
                        'CALC_QUAD_RISK', 'CALC_CALF_RISK', 'CALC_LOWBACK_RISK']
            risk_labels = ['Hamstring', 'Groin', 'Hip Flexor', 'Quad', 'Calf', 'Low Back']
            
            team_risks = []
            for col in risk_cols:
                if col in phys.columns:
                    team_risks.append(phys[col].mean())
                else:
                    team_risks.append(5)
            
            team_risks.append(team_risks[0])  # Close radar
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=team_risks,
                theta=risk_labels + [risk_labels[0]],
                fill='toself',
                fillcolor='rgba(239, 68, 68, 0.3)',
                line=dict(color='#EF4444', width=2),
                name='Team Avg'
            ))
            
            fig.update_layout(
                polar=dict(
                radialaxis=dict(
                    visible=True, 
                    range=[0, 10], 
                    tickfont=dict(color='#4B5563', size=10),
                    tickvals=[2, 4, 6, 8, 10],
                    gridcolor='#E5E7EB'
                ),
                angularaxis=dict(
                    tickfont=dict(color='#374151', size=11),
                        rotation=90,
                        direction='clockwise',
                        gridcolor='#E5E7EB'
                    ),
                bgcolor='white'
                ),
                paper_bgcolor='white',
                font=dict(color='#374151', size=12),
                height=350,
                margin=dict(l=80, r=80, t=60, b=60),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Risk Breakdown**")
            
            risk_df = pd.DataFrame({
                'Area': risk_labels,
                'Risk': team_risks[:-1]
            }).sort_values('Risk', ascending=False)
            
            colors = ['#10B981' if r < 4 else '#F59E0B' if r < 7 else '#EF4444' for r in risk_df['Risk']]
            
            fig = go.Figure(go.Bar(
                x=risk_df['Risk'],
                y=risk_df['Area'],
                orientation='h',
                marker_color=colors,
                text=[f"{r:.1f}" for r in risk_df['Risk']],
                textposition='outside'
            ))
            
            fig.update_layout(
                height=350,
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='#374151', size=12),
                xaxis=dict(range=[0, 10], title='Risk Score'),
                margin=dict(l=100, r=40, t=40, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # ===== ROW 6: WELLNESS COMPARISON =====
        if len(well) > 0 and 'Athlete' in well.columns:
            st.subheader("😴 Sleep & Wellness Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Sleep Efficiency (Quality × Duration)**")
                
                if 'SLEEP QUALITY' in well.columns and 'SLEEP DURATION' in well.columns:
                    sleep_eff = well.groupby('Athlete').apply(
                        lambda x: (x['SLEEP QUALITY'].mean() * x['SLEEP DURATION'].mean()) / 10
                    ).sort_values(ascending=True)
                    
                    colors = ['#10B981' if v >= 5 else '#F59E0B' if v >= 3.5 else '#EF4444' for v in sleep_eff.values]
                    
                    fig = go.Figure(go.Bar(
                        x=sleep_eff.values,
                        y=sleep_eff.index,
                        orientation='h',
                        marker_color=colors,
                        text=[f"{v:.1f}" for v in sleep_eff.values],
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        height=250,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=12),
                        xaxis=dict(title='Efficiency Score'),
                        margin=dict(l=100, r=40, t=10, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Stress vs Fatigue**")
                
                if 'STRESS LEVEL' in well.columns and 'FATIGUE LEVEL' in well.columns:
                    stress_fatigue = well.groupby('Athlete').agg({
                        'STRESS LEVEL': 'mean',
                        'FATIGUE LEVEL': 'mean'
                    }).reset_index()
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Stress',
                        x=stress_fatigue['Athlete'],
                        y=stress_fatigue['STRESS LEVEL'],
                        marker_color='#F59E0B'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Fatigue',
                        x=stress_fatigue['Athlete'],
                        y=stress_fatigue['FATIGUE LEVEL'],
                        marker_color='#EF4444'
                    ))
                    
                    fig.update_layout(
                        height=250,
                        barmode='group',
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=12),
                        yaxis=dict(range=[0, 10], title='Level'),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # ===== ROW 7: HIGH RISK ALERTS =====
        st.subheader("🚨 Alerts & Flags")
        
        alerts = []
        
        for athlete in athletes:
            ath_phys = phys[phys['Athlete'] == athlete]
            ath_well = well[well['Athlete'] == athlete] if len(well) > 0 and 'Athlete' in well.columns else pd.DataFrame()
            
            if len(ath_phys) > 0:
                latest = ath_phys.sort_values('Date').iloc[-1] if 'Date' in ath_phys.columns else ath_phys.iloc[-1]
                
                # Check for high risks
                for risk_col, risk_name in [('CALC_HAMSTRING_RISK', 'Hamstring'), ('CALC_GROIN_RISK', 'Groin'),
                                            ('CALC_QUAD_RISK', 'Quad'), ('CALC_CALF_RISK', 'Calf'),
                                            ('CALC_LOWBACK_RISK', 'Low Back')]:
                    if risk_col in latest and latest[risk_col] > 7:
                        alerts.append({'Athlete': athlete, 'Alert': f'High {risk_name} Risk', 
                                      'Value': f"{latest[risk_col]:.1f}", 'Type': 'risk'})
            
            if len(ath_well) > 0:
                latest_well = ath_well.sort_values('Date').iloc[-1] if 'Date' in ath_well.columns else ath_well.iloc[-1]
                
                # Check wellness
                if 'Wellness %' in latest_well and latest_well['Wellness %'] < 70:
                    alerts.append({'Athlete': athlete, 'Alert': 'Low Wellness', 
                                  'Value': f"{latest_well['Wellness %']:.0f}%", 'Type': 'wellness'})
                
                if 'FATIGUE LEVEL' in latest_well and latest_well['FATIGUE LEVEL'] > 7:
                    alerts.append({'Athlete': athlete, 'Alert': 'High Fatigue', 
                                  'Value': f"{latest_well['FATIGUE LEVEL']:.0f}/10", 'Type': 'fatigue'})
        
        if alerts:
            alert_df = pd.DataFrame(alerts)
            
            cols = st.columns(min(len(alerts), 4))
            for idx, alert in enumerate(alerts[:8]):  # Show max 8 alerts
                col_idx = idx % 4
                
                icon = '⚠️' if alert['Type'] == 'risk' else '😫' if alert['Type'] == 'fatigue' else '📉'
                color = '#EF4444' if alert['Type'] == 'risk' else '#F59E0B'
                
                with cols[col_idx]:
                    st.markdown(f"""
                    <div style="background-color: {color}22; border: 1px solid {color}; 
                                border-radius: 8px; padding: 10px; margin: 5px 0;">
                        <div style="font-size: 11px; color: #6B7280;">{alert['Athlete']}</div>
                        <div style="font-size: 14px; color: white; font-weight: bold;">{icon} {alert['Alert']}</div>
                        <div style="font-size: 16px; color: {color}; font-weight: bold;">{alert['Value']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("✅ No high-risk alerts at this time!")
        
        st.markdown("---")
        
        # ===== ROW 8: COMPREHENSIVE ATHLETE TABLE =====
        st.subheader("📋 Complete Athlete Summary")
        
        summary_data = []
        for athlete in athletes:
            ath_phys = phys[phys['Athlete'] == athlete]
            ath_well = well[well['Athlete'] == athlete] if len(well) > 0 and 'Athlete' in well.columns else pd.DataFrame()
            ath_info = info[info['Athlete'] == athlete] if len(info) > 0 and 'Athlete' in info.columns else pd.DataFrame()
            
            row = {'Athlete': athlete}
            
            # Physical metrics
            if len(ath_phys) > 0:
                row['Sessions'] = len(ath_phys)
                row['Total Time (hrs)'] = round(ath_phys['Move Time'].sum() / 60, 1) if 'Move Time' in ath_phys.columns else 0
                row['Total Distance (km)'] = round(ath_phys['Distance'].sum() / 1000, 1) if 'Distance' in ath_phys.columns else 0
                row['Avg Load'] = round(ath_phys['Total Load'].mean(), 0) if 'Total Load' in ath_phys.columns else 0
                
                latest = ath_phys.sort_values('Date').iloc[-1] if 'Date' in ath_phys.columns else ath_phys.iloc[-1]
                row['Volume STEN'] = round(latest.get('CALC_VOLUME_STEN', 0), 1)
                row['Intensity STEN'] = round(latest.get('CALC_INTENSITY_STEN', 0), 1)
            
            # Wellness metrics
            if len(ath_well) > 0:
                latest_well = ath_well.sort_values('Date').iloc[-1] if 'Date' in ath_well.columns else ath_well.iloc[-1]
                row['Wellness %'] = round(latest_well.get('Wellness %', 0), 0)
                row['Readiness'] = round(latest_well.get('READINESS TO TRAIN', 0), 1)
            
            # Info
            if len(ath_info) > 0:
                row['Sport'] = ath_info['SPORT'].iloc[0] if 'SPORT' in ath_info.columns else 'N/A'
                row['Risk Tier'] = ath_info['RISK TIER'].iloc[0] if 'RISK TIER' in ath_info.columns else 'N/A'
            
            summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 2: PHYSICAL
# ============================================================================

elif page == "🏃 Physical":
    st.title("Physical Performance Analysis")
    
    if 'physical' not in data:
        st.warning("📋 Paste Physical sheet URL in sidebar to view data")
    else:
        df = data['physical']
        
        # Filter by athlete
        if selected_athlete != "All Athletes":
            df = df[df['Athlete'] == selected_athlete]
        
        # Filter by date
        if date_range and len(date_range) == 2 and 'Date' in df.columns:
            df = df[(df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])]
        
        if len(df) == 0:
            st.warning("No data for selected filters")
        else:
            # Get latest row for cards
            if 'Date' in df.columns:
                latest = df.sort_values('Date', ascending=False).iloc[0]
            else:
                latest = df.iloc[-1]
            
            # ===== ROW 1: STEN SCORE CARDS =====
            st.subheader("STEN Scores (Latest)")
            col1, col2, col3, col4 = st.columns(4)
            
            sten_metrics = [
                ('CALC_VOLUME_STEN', 'Volume', col1),
                ('CALC_INTENSITY_STEN', 'Intensity', col2),
                ('CALC_SYMMETRY_STEN', 'Symmetry', col3),
                ('CALC_PERFORMANCE_STEN', 'Performance', col4)
            ]
            
            for col_name, label, col in sten_metrics:
                if col_name in latest:
                    val = latest[col_name]
                    with col:
                        st.markdown(f"""
                        <div class="sten-card">
                            <div class="sten-label">{label}</div>
                            <div class="sten-score">{val:.1f}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ===== ROW 2: SITE-SPECIFIC RISK =====
            st.subheader("Site-Specific Injury Risk (0-10)")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Radar Chart
                risk_metrics = {
                    'Hamstring': latest.get('CALC_HAMSTRING_RISK', 5),
                    'Groin': latest.get('CALC_GROIN_RISK', 5),
                    'Hip Flexor': latest.get('CALC_HIPFLEXOR_RISK', 5),
                    'Quad': latest.get('CALC_QUAD_RISK', 5),
                    'Calf': latest.get('CALC_CALF_RISK', 5),
                    'Low Back': latest.get('CALC_LOWBACK_RISK', 5)
                }
                
                categories = list(risk_metrics.keys())
                values = list(risk_metrics.values())
                values.append(values[0])  # Close the radar
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    fillcolor='rgba(239, 68, 68, 0.3)',
                    line=dict(color='#EF4444', width=2),
                    name='Risk'
                ))
                
                fig.update_layout(
                    polar=dict(
                    radialaxis=dict(
                        visible=True, 
                        range=[0, 10], 
                        tickfont=dict(color='#4B5563', size=10),
                        tickvals=[2, 4, 6, 8, 10],
                        gridcolor='#E5E7EB'
                    ),
                    angularaxis=dict(
                        tickfont=dict(color='#374151', size=11),
                            rotation=90,
                            direction='clockwise',
                            gridcolor='#E5E7EB'
                        ),
                    bgcolor='white'
                    ),
                    paper_bgcolor='white',
                    font=dict(color='#374151', size=12),
                    height=350,
                    margin=dict(l=80, r=80, t=60, b=60),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Horizontal Bar Chart
                risk_df = pd.DataFrame({
                    'Area': categories,
                    'Risk': [risk_metrics[c] for c in categories]
                }).sort_values('Risk', ascending=True)
                
                colors = ['#10B981' if r < 4 else '#F59E0B' if r < 7 else '#EF4444' for r in risk_df['Risk']]
                
                fig = go.Figure(go.Bar(
                    x=risk_df['Risk'],
                    y=risk_df['Area'],
                    orientation='h',
                    marker_color=colors,
                    text=risk_df['Risk'].round(1),
                    textposition='outside'
                ))
                
                fig.update_layout(
                    xaxis=dict(range=[0, 10], title='Risk Score'),
                    yaxis=dict(title=''),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='#374151', size=12),
                    height=350,
                    margin=dict(l=100, r=40, t=40, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # ===== ROW 3: VOLUME VS INTENSITY TREND =====
            st.subheader("Volume vs Intensity Trend")
            
            if 'Date' in df.columns:
                df_sorted = df.sort_values('Date')
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(
                        x=df_sorted['Date'], 
                        y=df_sorted['CALC_VOLUME_STEN'],
                        name='Volume STEN',
                        line=dict(color='#3B82F6', width=3)
                    ),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df_sorted['Date'], 
                        y=df_sorted['CALC_INTENSITY_STEN'],
                        name='Intensity STEN',
                        line=dict(color='#EF4444', width=3)
                    ),
                    secondary_y=False
                )
                
                # Add threshold lines
                fig.add_hline(y=7, line_dash="dash", line_color="orange", annotation_text="High")
                fig.add_hline(y=3, line_dash="dash", line_color="orange", annotation_text="Low")
                
                fig.update_layout(
                    height=350,
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='#374151', size=12),
                    yaxis=dict(range=[0, 10], title='STEN Score'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # ===== ROW 4: ACUTE:CHRONIC WORKLOAD RATIO =====
            st.subheader("📊 Acute:Chronic Workload Ratio (A:C)")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if 'Date' in df.columns and 'CALC_ACWR_LOAD' in df.columns:
                    df_sorted = df.sort_values('Date')
                    acwr_valid = df_sorted[df_sorted['CALC_ACWR_LOAD'].notna()]
                    
                    if len(acwr_valid) > 0:
                        fig = go.Figure()
                        
                        # A:C ratio line
                        fig.add_trace(go.Scatter(
                            x=acwr_valid['Date'],
                            y=acwr_valid['CALC_ACWR_LOAD'],
                            name='A:C Ratio',
                            line=dict(color='#A78BFA', width=3),
                            fill='tozeroy',
                            fillcolor='rgba(167, 139, 250, 0.2)'
                        ))
                        
                        # Optimal zone shading
                        fig.add_hrect(y0=0.8, y1=1.3, fillcolor="green", opacity=0.1, line_width=0)
                        fig.add_hrect(y0=1.3, y1=1.5, fillcolor="yellow", opacity=0.1, line_width=0)
                        fig.add_hrect(y0=1.5, y1=2.0, fillcolor="red", opacity=0.1, line_width=0)
                        fig.add_hrect(y0=0, y1=0.8, fillcolor="blue", opacity=0.1, line_width=0)
                        
                        # Reference lines
                        fig.add_hline(y=1.0, line_dash="dash", line_color="#9CA3AF", opacity=0.7)
                        fig.add_hline(y=1.3, line_dash="dot", line_color="#F59E0B", opacity=0.7)
                        fig.add_hline(y=1.5, line_dash="dot", line_color="#EF4444", opacity=0.7)
                        
                        fig.update_layout(
                            height=300,
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                            font=dict(color='#374151', size=12),
                            yaxis=dict(range=[0.5, 2.0], title='A:C Ratio'),
                            xaxis=dict(title=''),
                            showlegend=False,
                            margin=dict(l=40, r=40, t=20, b=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough data for A:C ratio (need 7+ days)")
            
            with col2:
                # Current A:C status card
                if 'CALC_ACWR_LOAD' in df.columns:
                    acwr_data = df.sort_values('Date')['CALC_ACWR_LOAD'].dropna() if 'Date' in df.columns else df['CALC_ACWR_LOAD'].dropna()
                    
                    if len(acwr_data) > 0:
                        current_acwr = acwr_data.iloc[-1]
                    else:
                        current_acwr = 1.0
                    
                    if pd.isna(current_acwr):
                        current_acwr = 1.0
                    
                    if current_acwr < 0.8:
                        status = "UNDERTRAINED"
                        status_color = "#3B82F6"
                        status_desc = "Load too low, fitness may decrease"
                    elif current_acwr <= 1.3:
                        status = "OPTIMAL"
                        status_color = "#10B981"
                        status_desc = "Ideal training zone"
                    elif current_acwr <= 1.5:
                        status = "CAUTION"
                        status_color = "#F59E0B"
                        status_desc = "Monitor closely, increased risk"
                    else:
                        status = "HIGH RISK"
                        status_color = "#EF4444"
                        status_desc = "Overload! Reduce training"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {status_color}cc, {status_color}); 
                                padding: 20px; border-radius: 12px; text-align: center;">
                        <div style="font-size: 12px; color: white; text-transform: uppercase; opacity: 0.9;">Current A:C Ratio</div>
                        <div style="font-size: 42px; font-weight: bold; color: white;">{current_acwr:.2f}</div>
                        <div style="font-size: 16px; color: white; font-weight: bold;">{status}</div>
                        <div style="font-size: 11px; color: white; opacity: 0.8; margin-top: 5px;">{status_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Acute vs Chronic values
                    if 'CALC_ACUTE_LOAD' in df.columns and 'CALC_CHRONIC_LOAD' in df.columns:
                        df_sorted_ac = df.sort_values('Date') if 'Date' in df.columns else df
                        acute_data = df_sorted_ac['CALC_ACUTE_LOAD'].dropna()
                        chronic_data = df_sorted_ac['CALC_CHRONIC_LOAD'].dropna()
                        
                        acute = acute_data.iloc[-1] if len(acute_data) > 0 else 0
                        chronic = chronic_data.iloc[-1] if len(chronic_data) > 0 else 0
                        
                        st.markdown(f"""
                        <div style="display: flex; gap: 10px; margin-top: 10px;">
                            <div style="flex: 1; background-color: #F3F4F6; padding: 10px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 10px; color: #6B7280;">7-Day (Acute)</div>
                                <div style="font-size: 18px; font-weight: bold; color: #60A5FA;">{acute:,.0f}</div>
                            </div>
                            <div style="flex: 1; background-color: #F3F4F6; padding: 10px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 10px; color: #6B7280;">28-Day (Chronic)</div>
                                <div style="font-size: 18px; font-weight: bold; color: #A78BFA;">{chronic:,.0f}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ===== ROW 5: WORKLOAD WITH RISK MULTIPLIER =====
            st.subheader("Workload with Risk Multiplier")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Date' in df.columns:
                    df_sorted = df.sort_values('Date')
                    
                    # Calculate risk multiplier based on average risk
                    avg_risk = df_sorted[['CALC_HAMSTRING_RISK', 'CALC_GROIN_RISK', 'CALC_QUAD_RISK', 
                                         'CALC_CALF_RISK', 'CALC_LOWBACK_RISK']].mean(axis=1)
                    risk_multiplier = 1 + (avg_risk / 10) * 0.5  # 1.0 to 1.5x
                    
                    workload_adjusted = df_sorted['CALC_WORKLOAD_STEN'] * risk_multiplier
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=df_sorted['Date'],
                        y=df_sorted['CALC_WORKLOAD_STEN'],
                        name='Base Workload',
                        marker_color='#3B82F6'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df_sorted['Date'],
                        y=workload_adjusted,
                        name='Risk-Adjusted',
                        line=dict(color='#F59E0B', width=3)
                    ))
                    
                    fig.update_layout(
                        height=300,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=12),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=60, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Asymmetry Tracking
                st.markdown("**Asymmetry Tracking**")
                
                if 'Date' in df.columns:
                    df_sorted = df.sort_values('Date')
                    
                    fig = go.Figure()
                    
                    # Plot key asymmetry metrics
                    asym_cols_plot = ['Total Athlete g-Load Asymmetry', 'Total Impact Asymmetry']
                    colors = ['#A78BFA', '#34D399']
                    
                    for col, color in zip(asym_cols_plot, colors):
                        if col in df_sorted.columns:
                            fig.add_trace(go.Scatter(
                                x=df_sorted['Date'],
                                y=df_sorted[col].abs(),
                                name=col.replace(' Asymmetry', ''),
                                line=dict(color=color, width=2)
                            ))
                    
                    # Add threshold zone
                    fig.add_hrect(y0=0, y1=5, fillcolor="green", opacity=0.1, line_width=0)
                    fig.add_hrect(y0=5, y1=10, fillcolor="yellow", opacity=0.1, line_width=0)
                    fig.add_hrect(y0=10, y1=20, fillcolor="red", opacity=0.1, line_width=0)
                    
                    fig.update_layout(
                        height=300,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=12),
                        yaxis=dict(title='Asymmetry %'),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=60, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # ===== ROW 6: HEATMAPS (Training Load & Maximum Speed) =====
            st.subheader("Training Heatmaps")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Training Load**")
                if 'Date' in df.columns and 'Total Load' in df.columns:
                    df_heat = df.copy()
                    df_heat['DayOfWeek'] = df_heat['Date'].dt.day_name()
                    df_heat['WeekNum'] = df_heat['Date'].dt.isocalendar().week
                    
                    pivot_load = df_heat.pivot_table(
                        values='Total Load',
                        index='DayOfWeek',
                        columns='WeekNum',
                        aggfunc='mean'
                    )
                    
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    pivot_load = pivot_load.reindex([d for d in day_order if d in pivot_load.index])
                    
                    # Replace 0 with NaN so they show as blank
                    pivot_load = pivot_load.replace(0, np.nan)
                    
                    # Create text array - blank for NaN, value for actual data
                    text_vals = np.where(pd.isna(pivot_load.values), '', np.round(pivot_load.values, 0).astype(str))
                    
                    fig_load = go.Figure(data=go.Heatmap(
                        z=pivot_load.values,
                        x=[f'W{int(w)}' for w in pivot_load.columns],
                        y=pivot_load.index,
                        colorscale='Blues',
                        showscale=False,
                        text=text_vals,
                        texttemplate="%{text}",
                        textfont=dict(size=14, color='black'),
                        hovertemplate='%{y}<br>%{x}<br>Load: %{z:.0f}<extra></extra>',
                        xgap=2,
                        ygap=2
                    ))
                    
                    fig_load.update_layout(
                        height=300,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=12),
                        xaxis=dict(side='top', tickfont=dict(color='#4B5563')),
                        yaxis=dict(tickfont=dict(color='#4B5563')),
                        margin=dict(l=100, r=20, t=60, b=40)
                    )
                    
                    st.plotly_chart(fig_load, use_container_width=True)
                else:
                    st.info("Need 'Date' and 'Total Load' columns for heatmap")
            
            with col2:
                st.markdown("**Maximum Speed**")
                if 'Date' in df.columns and 'Maximum Speed' in df.columns:
                    df_heat = df.copy()
                    df_heat['DayOfWeek'] = df_heat['Date'].dt.day_name()
                    df_heat['WeekNum'] = df_heat['Date'].dt.isocalendar().week
                    
                    pivot_speed = df_heat.pivot_table(
                        values='Maximum Speed',
                        index='DayOfWeek',
                        columns='WeekNum',
                        aggfunc='mean'
                    )
                    
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    pivot_speed = pivot_speed.reindex([d for d in day_order if d in pivot_speed.index])
                    
                    # Replace 0 with NaN so they show as blank
                    pivot_speed = pivot_speed.replace(0, np.nan)
                    
                    # Create text array - blank for NaN, value for actual data
                    text_vals = np.where(pd.isna(pivot_speed.values), '', np.round(pivot_speed.values, 1).astype(str))
                    
                    fig_speed = go.Figure(data=go.Heatmap(
                        z=pivot_speed.values,
                        x=[f'W{int(w)}' for w in pivot_speed.columns],
                        y=pivot_speed.index,
                        colorscale='Reds',
                        showscale=False,
                        text=text_vals,
                        texttemplate="%{text}",
                        textfont=dict(size=14, color='black'),
                        hovertemplate='%{y}<br>%{x}<br>Speed: %{z:.1f} m/s<extra></extra>',
                        xgap=2,
                        ygap=2
                    ))
                    
                    fig_speed.update_layout(
                        height=300,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=12),
                        xaxis=dict(side='top', tickfont=dict(color='#4B5563')),
                        yaxis=dict(tickfont=dict(color='#4B5563')),
                        margin=dict(l=100, r=20, t=60, b=40)
                    )
                    
                    st.plotly_chart(fig_speed, use_container_width=True)
                else:
                    st.info("Need 'Date' and 'Maximum Speed' columns for heatmap")
            
            st.markdown("---")
            
            # ===== ROW 7: CUSTOM DATA TABLE =====
            st.subheader("📋 Data Explorer")
            
            # Define column categories for easier selection
            column_categories = {
                'Basic Info': ['Athlete', 'Date', 'Session', 'Session Title'],
                'Volume Metrics': ['Distance', 'Total Load', 'Move Time', '# ACCELS', '# DECELS'],
                'Intensity Metrics': ['Athlete Intensity', 'PEAK ACCEL', 'PEAK DECEL', 'Maximum Speed', 'Impact Intensity'],
                'Speed Metrics': ['Maximum Speed', 'Distance at High Speed', 'Distance at Very High Speed'],
                'Impact Metrics': ['Total Impact', 'Total Vertical Impact', 'Maximum Jump Height'],
                'Asymmetry Metrics': [c for c in df.columns if 'Asymmetry' in c],
                'Calculated STEN': ['CALC_VOLUME_STEN', 'CALC_INTENSITY_STEN', 'CALC_SYMMETRY_STEN', 'CALC_PERFORMANCE_STEN', 'CALC_WORKLOAD_STEN'],
                'Calculated Risks': ['CALC_HAMSTRING_RISK', 'CALC_GROIN_RISK', 'CALC_HIPFLEXOR_RISK', 'CALC_QUAD_RISK', 'CALC_CALF_RISK', 'CALC_LOWBACK_RISK'],
                'A:C Ratio': ['CALC_ACWR_LOAD', 'CALC_ACUTE_LOAD', 'CALC_CHRONIC_LOAD']
            }
            
            # Filter to only columns that exist in the dataframe
            available_columns = []
            for category, cols in column_categories.items():
                existing = [c for c in cols if c in df.columns]
                if existing:
                    available_columns.extend(existing)
            
            # Add any remaining columns not in categories
            other_cols = [c for c in df.columns if c not in available_columns and not c.startswith('CALC_') and '_z' not in c.lower() and '_abs' not in c.lower()]
            available_columns.extend(other_cols)
            
            # Remove duplicates while preserving order
            available_columns = list(dict.fromkeys(available_columns))
            
            # Default columns to show
            default_cols = ['Athlete', 'Date', 'Total Load', 'Maximum Speed', 'CALC_VOLUME_STEN', 'CALC_INTENSITY_STEN']
            default_selection = [c for c in default_cols if c in available_columns]
            
            # Multiselect for columns
            selected_columns = st.multiselect(
                "Select columns to display:",
                options=available_columns,
                default=default_selection,
                help="Choose which columns to show in the table below"
            )
            
            if selected_columns:
                # Create display dataframe
                display_df = df[selected_columns].copy()
                
                # Format date column if present
                if 'Date' in display_df.columns:
                    display_df = display_df.sort_values('Date', ascending=False)
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                
                # Round numeric columns
                for col in display_df.select_dtypes(include=[np.number]).columns:
                    display_df[col] = display_df[col].round(2)
                
                st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
                
                # Quick stats for selected numeric columns
                numeric_selected = [c for c in selected_columns if c in df.select_dtypes(include=[np.number]).columns]
                if numeric_selected and len(numeric_selected) <= 6:
                    st.markdown("**Quick Stats:**")
                    stat_cols = st.columns(len(numeric_selected))
                    for idx, col in enumerate(numeric_selected):
                        with stat_cols[idx]:
                            st.metric(
                                label=col[:20] + "..." if len(col) > 20 else col,
                                value=f"{df[col].mean():.1f}",
                                delta=f"σ {df[col].std():.1f}"
                            )
            else:
                st.info("Select at least one column to display data")

# ============================================================================
# PAGE 3: WELLNESS
# ============================================================================

elif page == "❤️ Wellness":
    st.title("Wellness Monitoring")
    
    if 'wellness' not in data:
        st.warning("📋 Paste Wellness sheet URL in sidebar to view data")
    else:
        df = data['wellness'].copy()
        
        # Standardize name column if needed
        if 'Name' in df.columns and 'Athlete' not in df.columns:
            df = df.rename(columns={'Name': 'Athlete'})
        
        # Filter by athlete
        if selected_athlete != "All Athletes" and 'Athlete' in df.columns:
            df = df[df['Athlete'] == selected_athlete]
        
        # Filter by date
        if date_range and len(date_range) == 2 and 'Date' in df.columns:
            df = df[(df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])]
        
        if len(df) == 0:
            st.warning("No data for selected filters")
        else:
            # Get latest row
            if 'Date' in df.columns:
                latest = df.sort_values('Date', ascending=False).iloc[0]
            else:
                latest = df.iloc[-1]
            
            # ===== ROW 1: WELLNESS STATUS BANNER =====
            # Calculate Wellness % from components
            sleep_dur = float(latest.get('SLEEP DURATION', 0) or 0)
            sleep_qual = float(latest.get('SLEEP QUALITY', 0) or 0)
            energy = float(latest.get('ENERGY LEVEL', 0) or 0)
            fatigue = float(latest.get('FATIGUE LEVEL', 0) or 0)
            stress = float(latest.get('STRESS LEVEL', 0) or 0)
            readiness = float(latest.get('READINESS TO TRAIN', 0) or 0)

            # Soreness levels (4 areas)
            soreness_total = sum([
                float(latest.get('Soreness Level 1', 0) or 0),
                float(latest.get('Soreness Level 2', 0) or 0),
                float(latest.get('Soreness Level 3', 0) or 0),
                float(latest.get('Soreness Level 4', 0) or 0)
            ])

            # Pain levels (4 areas)
            pain_total = sum([
                float(latest.get('Pain Level 1', 0) or 0),
                float(latest.get('Pain Level 2', 0) or 0),
                float(latest.get('Pain Level 3', 0) or 0),
                float(latest.get('Pain Level 4', 0) or 0)
            ])

            # Formula: 50% core metrics + 20% soreness + 30% pain
            wellness_metrics = (min(10, (sleep_dur/8)*10) + sleep_qual + energy + (10-fatigue) + (10-stress) + readiness) / 60 * 0.5
            soreness_factor = (1 - (min(10, soreness_total)/10)) * 0.2
            pain_factor = (1 - (min(10, pain_total)/10)) * 0.3

            wellness_pct = round((wellness_metrics + soreness_factor + pain_factor) * 100, 1)
            
            # Determine color based on wellness score
            if wellness_pct >= 85:
                color = 'GREEN'
            elif wellness_pct >= 70:
                color = 'YELLOW'
            else:
                color = 'RED'
            
            color_map = {'GREEN': '#10B981', 'YELLOW': '#F59E0B', 'RED': '#EF4444'}
            bg_color = color_map.get(color, '#F59E0B')
            
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;">
                <div style="font-size: 48px; font-weight: bold; color: white;">{wellness_pct}%</div>
                <div style="font-size: 18px; color: white; text-transform: uppercase;">Wellness Score - {color}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # ===== ROW 2: LATEST METRIC CARDS =====
            st.subheader("Latest Wellness Metrics")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            metrics = [
                ('SLEEP DURATION', 'Sleep Hrs', col1, '#3B82F6'),
                ('SLEEP QUALITY', 'Sleep Quality', col2, '#8B5CF6'),
                ('ENERGY LEVEL', 'Energy', col3, '#10B981'),
                ('FATIGUE LEVEL', 'Fatigue', col4, '#EF4444'),
                ('STRESS LEVEL', 'Stress', col5, '#F59E0B'),
                ('READINESS TO TRAIN', 'Readiness', col6, '#06B6D4')
            ]
            
            for col_name, label, col, color in metrics:
                if col_name in latest:
                    val = latest[col_name]
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value" style="color: {color};">{val}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ===== ROW 3: SLEEP & READINESS TRENDS =====
            st.subheader("Trends")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Sleep Duration & Quality**")
                if 'Date' in df.columns:
                    df_sorted = df.sort_values('Date')
                    
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    if 'SLEEP DURATION' in df_sorted.columns:
                        fig.add_trace(
                            go.Bar(
                                x=df_sorted['Date'],
                                y=df_sorted['SLEEP DURATION'],
                                name='Duration (hrs)',
                                marker_color='#3B82F6',
                                opacity=0.7
                            ),
                            secondary_y=False
                        )
                    
                    if 'SLEEP QUALITY' in df_sorted.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df_sorted['Date'],
                                y=df_sorted['SLEEP QUALITY'],
                                name='Quality',
                                line=dict(color='#8B5CF6', width=3)
                            ),
                            secondary_y=True
                        )
                    
                    fig.update_layout(
                        height=300,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=12),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    fig.update_yaxes(title_text="Hours", secondary_y=False)
                    fig.update_yaxes(title_text="Quality (1-10)", range=[0, 10], secondary_y=True)
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Energy, Fatigue & Readiness**")
                if 'Date' in df.columns:
                    df_sorted = df.sort_values('Date')
                    
                    fig = go.Figure()
                    
                    trend_metrics = [
                        ('ENERGY LEVEL', 'Energy', '#10B981'),
                        ('FATIGUE LEVEL', 'Fatigue', '#EF4444'),
                        ('READINESS TO TRAIN', 'Readiness', '#06B6D4')
                    ]
                    
                    for col_name, label, color in trend_metrics:
                        if col_name in df_sorted.columns:
                            fig.add_trace(go.Scatter(
                                x=df_sorted['Date'],
                                y=df_sorted[col_name],
                                name=label,
                                line=dict(color=color, width=2)
                            ))
                    
                    fig.update_layout(
                        height=300,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=12),
                        yaxis=dict(range=[0, 10], title='Score (1-10)'),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # ===== ROW 4: ACUTE:CHRONIC WELLNESS RATIO =====
            st.subheader("Acute:Chronic Wellness Ratio (ACWR)")
            
            if 'Date' in df.columns and 'Wellness %' in df.columns:
                # Need enough data for rolling calculations
                df_acwr = df.sort_values('Date').copy()
                
                # Calculate rolling averages per athlete
                if selected_athlete != "All Athletes":
                    df_acwr['Acute_7d'] = df_acwr['Wellness %'].rolling(window=7, min_periods=1).mean()
                    df_acwr['Chronic_28d'] = df_acwr['Wellness %'].rolling(window=28, min_periods=7).mean()
                else:
                    # Group by athlete for rolling
                    df_acwr['Acute_7d'] = df_acwr.groupby('Athlete')['Wellness %'].transform(
                        lambda x: x.rolling(window=7, min_periods=1).mean()
                    )
                    df_acwr['Chronic_28d'] = df_acwr.groupby('Athlete')['Wellness %'].transform(
                        lambda x: x.rolling(window=28, min_periods=7).mean()
                    )
                
                df_acwr['ACWR'] = df_acwr['Acute_7d'] / df_acwr['Chronic_28d']
                df_acwr['ACWR'] = df_acwr['ACWR'].replace([np.inf, -np.inf], np.nan)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = go.Figure()
                    
                    # ACWR line
                    fig.add_trace(go.Scatter(
                        x=df_acwr['Date'],
                        y=df_acwr['ACWR'],
                        name='ACWR',
                        line=dict(color='#A78BFA', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(167, 139, 250, 0.2)'
                    ))
                    
                    # Optimal zone
                    fig.add_hrect(y0=0.8, y1=1.3, fillcolor="green", opacity=0.1, 
                                  annotation_text="Optimal", annotation_position="top left")
                    fig.add_hrect(y0=1.3, y1=1.5, fillcolor="yellow", opacity=0.1,
                                  annotation_text="Caution", annotation_position="top left")
                    fig.add_hrect(y0=1.5, y1=2.0, fillcolor="red", opacity=0.1,
                                  annotation_text="High Risk", annotation_position="top left")
                    
                    fig.update_layout(
                        height=300,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=12),
                        yaxis=dict(range=[0.5, 2.0], title='ACWR'),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Current ACWR status
                    current_acwr = df_acwr['ACWR'].dropna().iloc[-1] if len(df_acwr['ACWR'].dropna()) > 0 else 1.0
                    
                    if current_acwr < 0.8:
                        status = "UNDERTRAINED"
                        status_color = "#3B82F6"
                    elif current_acwr <= 1.3:
                        status = "OPTIMAL"
                        status_color = "#10B981"
                    elif current_acwr <= 1.5:
                        status = "CAUTION"
                        status_color = "#F59E0B"
                    else:
                        status = "HIGH RISK"
                        status_color = "#EF4444"
                    
                    st.markdown(f"""
                    <div style="background-color: {status_color}; padding: 30px; border-radius: 12px; text-align: center; height: 100%;">
                        <div style="font-size: 14px; color: white; text-transform: uppercase; opacity: 0.9;">Current ACWR</div>
                        <div style="font-size: 48px; font-weight: bold; color: white;">{current_acwr:.2f}</div>
                        <div style="font-size: 18px; color: white; font-weight: bold;">{status}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div style="font-size: 11px; color: #6B7280; margin-top: 10px;">
                    <b>ACWR Guide:</b><br>
                    &lt;0.8 = Undertrained<br>
                    0.8-1.3 = Optimal<br>
                    1.3-1.5 = Caution<br>
                    &gt;1.5 = High Risk
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ===== ROW 5: WEEKLY WELLNESS HEATMAP =====
            st.subheader("Weekly Wellness Pattern")
            
            if 'Date' in df.columns and 'Day' in df.columns and 'Wellness %' in df.columns:
                df_heat = df.copy()
                df_heat['Week'] = df_heat.get('Week', df_heat['Date'].dt.isocalendar().week)
                
                pivot = df_heat.pivot_table(
                    values='Wellness %',
                    index='Day',
                    columns='Week',
                    aggfunc='mean'
                )
                
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                pivot = pivot.reindex([d for d in day_order if d in pivot.index])
                
                # Replace 0 with NaN so they show as blank
                pivot = pivot.replace(0, np.nan)
                
                # Create text array - blank for NaN, value for actual data
                text_vals = np.where(pd.isna(pivot.values), '', np.round(pivot.values, 0).astype(str))
                
                fig = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=[f'W{int(w)}' for w in pivot.columns],
                    y=pivot.index,
                    colorscale='RdYlGn',
                    showscale=False,
                    zmin=60,
                    zmax=100,
                    text=text_vals,
                    texttemplate="%{text}",
                    textfont=dict(size=11, color='#374151'),
                    hovertemplate='%{y}<br>%{x}<br>Wellness: %{z:.0f}%<extra></extra>',
                    xgap=2,
                    ygap=2
                ))
                
                fig.update_layout(
                    height=280,
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='#374151', size=12),
                    xaxis=dict(side='top', tickfont=dict(color='#4B5563')),
                    yaxis=dict(tickfont=dict(color='#4B5563')),
                    margin=dict(l=100, r=40, t=60, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # ===== ROW 6: SORENESS & PAIN TRACKING =====
            st.subheader("Soreness & Pain Monitoring")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Current Soreness**")
                
                # Extract soreness data from latest
                soreness_data = []
                for i in range(1, 5):
                    area_col = f'Soreness Area {i}'
                    level_col = f'Soreness Level {i}'
                    
                    if area_col in latest and level_col in latest:
                        area = latest[area_col]
                        level = latest[level_col]
                        
                        if pd.notna(area) and pd.notna(level) and str(area).strip() != '':
                            soreness_data.append({'Area': area, 'Level': level, 'Type': 'Soreness'})
                
                if soreness_data:
                    sore_df = pd.DataFrame(soreness_data)
                    colors = ['#10B981' if l <= 3 else '#F59E0B' if l <= 6 else '#EF4444' for l in sore_df['Level']]
                    
                    fig = go.Figure(go.Bar(
                        x=sore_df['Level'],
                        y=sore_df['Area'],
                        orientation='h',
                        marker_color=colors,
                        text=sore_df['Level'],
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        xaxis=dict(range=[0, 10], title='Level'),
                        yaxis=dict(title=''),
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=12),
                        height=200,
                        margin=dict(l=100, r=40, t=20, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No soreness reported")
            
            with col2:
                st.markdown("**Current Pain**")
                
                # Extract pain data from latest
                pain_data = []
                for i in range(1, 5):
                    area_col = f'Pain Area {i}'
                    level_col = f'Pain Level {i}'
                    
                    if area_col in latest and level_col in latest:
                        area = latest[area_col]
                        level = latest[level_col]
                        
                        if pd.notna(area) and pd.notna(level) and str(area).strip() != '':
                            pain_data.append({'Area': area, 'Level': level, 'Type': 'Pain'})
                
                if pain_data:
                    pain_df = pd.DataFrame(pain_data)
                    colors = ['#10B981' if l <= 3 else '#F59E0B' if l <= 6 else '#EF4444' for l in pain_df['Level']]
                    
                    fig = go.Figure(go.Bar(
                        x=pain_df['Level'],
                        y=pain_df['Area'],
                        orientation='h',
                        marker_color=colors,
                        text=pain_df['Level'],
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        xaxis=dict(range=[0, 10], title='Level'),
                        yaxis=dict(title=''),
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=12),
                        height=200,
                        margin=dict(l=100, r=40, t=20, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No pain reported")
            
            st.markdown("---")
            
            # ===== ROW 7: WELLNESS TREND WITH STATUS =====
            st.subheader("Wellness % Trend")
            
            if 'Date' in df.columns and 'Wellness %' in df.columns:
                df_sorted = df.sort_values('Date')
                
                # Color code by status
                colors = df_sorted['COLOR'].map({'GREEN': '#10B981', 'YELLOW': '#F59E0B', 'RED': '#EF4444'}).fillna('#F59E0B')
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df_sorted['Date'],
                    y=df_sorted['Wellness %'],
                    mode='lines+markers',
                    line=dict(color='#60A5FA', width=2),
                    marker=dict(size=10, color=colors.tolist(), line=dict(width=2, color='white')),
                    name='Wellness %'
                ))
                
                if 'ROLLING WELLNESS' in df_sorted.columns:
                    fig.add_trace(go.Scatter(
                        x=df_sorted['Date'],
                        y=df_sorted['ROLLING WELLNESS'],
                        mode='lines',
                        line=dict(color='#A78BFA', width=2, dash='dash'),
                        name='Rolling Avg'
                    ))
                
                # Threshold lines
                fig.add_hline(y=85, line_dash="dash", line_color="green", annotation_text="Good (85)")
                fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Caution (70)")
                
                fig.update_layout(
                    height=300,
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='#374151', size=12),
                    yaxis=dict(range=[50, 100], title='Wellness %'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: INJURY
# ============================================================================

elif page == "🩹 Injury":
    st.title("Injury History & Impact")
    
    if 'injury' not in data:
        st.warning("📋 Paste Injury sheet URL in sidebar to view data")
    else:
        df = data['injury'].copy()
        
        # Standardize name column
        if 'Name' in df.columns and 'Athlete' not in df.columns:
            df = df.rename(columns={'Name': 'Athlete'})
        
        # Filter out empty rows
        df = df[df['Athlete'].notna() & (df['Athlete'] != '')]
        
        # Filter by athlete
        if selected_athlete != "All Athletes" and 'Athlete' in df.columns:
            df = df[df['Athlete'] == selected_athlete]
        
        if len(df) == 0:
            st.warning("No injury data for selected filters")
        else:
            # Get athlete summary (first row has aggregated data)
            athlete_summary = df.iloc[0]
            
            # ===== ROW 1: FINAL INJURY IMPACT SCORE BANNER =====
            final_score = athlete_summary.get('FINAL INJURY IMPACT SCORE', 0)
            
            # Determine risk level
            if final_score < 20:
                risk_color = '#10B981'
                risk_label = 'LOW RISK'
            elif final_score < 40:
                risk_color = '#F59E0B'
                risk_label = 'MODERATE RISK'
            else:
                risk_color = '#EF4444'
                risk_label = 'HIGH RISK'
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {risk_color}cc, {risk_color}); 
                        padding: 30px; border-radius: 16px; text-align: center; margin-bottom: 20px;
                        border: 2px solid {risk_color};">
                <div style="font-size: 14px; color: white; text-transform: uppercase; letter-spacing: 2px;">Final Injury Impact Score</div>
                <div style="font-size: 64px; font-weight: bold; color: white;">{final_score:.1f}</div>
                <div style="font-size: 20px; color: white; font-weight: bold;">{risk_label}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # ===== ROW 2: INJURY COUNT CARDS =====
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            cards = [
                ('TOTAL INJURIES', 'Total', col1, '#3B82F6'),
                ('DIRECT INJURIES', 'Direct', col2, '#8B5CF6'),
                ('ADJ INJURIES', 'Adjacent', col3, '#06B6D4'),
                ('SIGNIFICANT INJURIES', 'Significant', col4, '#EF4444'),
                ('TORSO', 'Torso', col5, '#F59E0B'),
                ('UPPER', 'Upper', col6, '#10B981')
            ]
            
            for col_name, label, col, color in cards:
                val = athlete_summary.get(col_name, 0)
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="color: {color};">{int(val) if pd.notna(val) else 0}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ===== ROW 3: MULTIPLIERS =====
            st.subheader("Risk Multipliers")
            col1, col2, col3, col4 = st.columns(4)
            
            multipliers = [
                ('KINETIC CHAIN MULTIPLIER', 'Kinetic Chain', col1),
                ('AGE MULTIPLIER', 'Age', col2),
                ('TOTAL MULTIPLIER', 'Total', col3),
                ('# OF CLUSTER SITES', 'Cluster Sites', col4)
            ]
            
            for col_name, label, col in multipliers:
                val = athlete_summary.get(col_name, 1)
                with col:
                    # Color based on multiplier value
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        if val <= 1.2:
                            mult_color = '#10B981'
                        elif val <= 1.5:
                            mult_color = '#F59E0B'
                        else:
                            mult_color = '#EF4444'
                        display_val = f"{val:.2f}" if col_name != '# OF CLUSTER SITES' else f"{int(val)}"
                    else:
                        mult_color = '#6B7280'
                        display_val = 'N/A'
                    
                    st.markdown(f"""
                    <div style="background-color: #F3F4F6; padding: 20px; border-radius: 10px; text-align: center; border-left: 4px solid {mult_color};">
                        <div style="font-size: 11px; color: #6B7280; text-transform: uppercase;">{label}</div>
                        <div style="font-size: 28px; font-weight: bold; color: {mult_color};">{display_val}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ===== ROW 4: INJURY BREAKDOWN CHARTS =====
            st.subheader("Injury Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**By Body Region**")
                if 'BODY REGION' in df.columns:
                    region_counts = df['BODY REGION'].value_counts().reset_index()
                    region_counts.columns = ['Region', 'Count']
                    region_counts = region_counts[region_counts['Region'].notna() & (region_counts['Region'] != '')]
                    
                    if len(region_counts) > 0:
                        fig = go.Figure(data=[go.Pie(
                            labels=region_counts['Region'],
                            values=region_counts['Count'],
                            hole=0.4,
                            marker=dict(colors=px.colors.qualitative.Set2),
                            textinfo='value+percent',
                            textfont=dict(size=11, color='#374151'),
                            textposition='outside'
                        )])
                        
                        fig.update_layout(
                            height=300,
                            paper_bgcolor='white',
                            font=dict(color='#374151', size=12),
                            margin=dict(l=20, r=20, t=20, b=40),
                            showlegend=True,
                            legend=dict(font=dict(size=11, color='#374151'))
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**By Tissue Type**")
                if 'Tissue Type' in df.columns:
                    tissue_counts = df['Tissue Type'].value_counts().reset_index()
                    tissue_counts.columns = ['Tissue', 'Count']
                    tissue_counts = tissue_counts[tissue_counts['Tissue'].notna() & (tissue_counts['Tissue'] != '')]
                    
                    if len(tissue_counts) > 0:
                        tissue_colors = {
                            'MUSCLE': '#EF4444',
                            'CARTILAGE': '#3B82F6',
                            'TENDON': '#F59E0B',
                            'LIGAMENT': '#10B981',
                            'BONE': '#8B5CF6',
                            'NEURAL': '#EC4899'
                        }
                        colors = [tissue_colors.get(t.upper(), '#6B7280') for t in tissue_counts['Tissue']]
                        
                        fig = go.Figure(go.Bar(
                            x=tissue_counts['Tissue'],
                            y=tissue_counts['Count'],
                            marker_color=colors,
                            text=tissue_counts['Count'],
                            textposition='outside'
                        ))
                        
                        fig.update_layout(
                            height=300,
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                            font=dict(color='#374151', size=12),
                            xaxis=dict(title=''),
                            yaxis=dict(title='Count'),
                            margin=dict(l=40, r=40, t=20, b=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # ===== ROW 5: KINETIC CHAIN & SURGERY =====
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**By Kinetic Chain**")
                if 'Kinteic Chain' in df.columns:
                    chain_counts = df['Kinteic Chain'].value_counts().reset_index()
                    chain_counts.columns = ['Chain', 'Count']
                    chain_counts = chain_counts[chain_counts['Chain'].notna() & (chain_counts['Chain'] != '')]
                    
                    if len(chain_counts) > 0:
                        chain_colors = {
                            'DIRECT': '#EF4444',
                            'ADJECENT': '#F59E0B',
                            'ADJACENT': '#F59E0B',
                            'TORSO': '#3B82F6',
                            'UPPER': '#8B5CF6',
                            'DISTAL': '#10B981',
                            'MID': '#06B6D4',
                            'PROXIMAL': '#EC4899'
                        }
                        colors = [chain_colors.get(c.upper(), '#6B7280') for c in chain_counts['Chain']]
                        
                        fig = go.Figure(go.Bar(
                            x=chain_counts['Count'],
                            y=chain_counts['Chain'],
                            orientation='h',
                            marker_color=colors,
                            text=chain_counts['Count'],
                            textposition='outside'
                        ))
                        
                        fig.update_layout(
                            height=250,
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                            font=dict(color='#374151', size=12),
                            xaxis=dict(title='Count'),
                            yaxis=dict(title=''),
                            margin=dict(l=80, r=40, t=20, b=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**By Surgery Type**")
                if 'Surgery' in df.columns:
                    surgery_counts = df['Surgery'].value_counts().reset_index()
                    surgery_counts.columns = ['Surgery', 'Count']
                    surgery_counts = surgery_counts[surgery_counts['Surgery'].notna() & (surgery_counts['Surgery'] != '')]
                    
                    if len(surgery_counts) > 0:
                        surgery_colors = {
                            'NONE': '#10B981',
                            'MINOR': '#F59E0B',
                            'MAJOR': '#EF4444',
                            'REVISION': '#8B5CF6'
                        }
                        colors = [surgery_colors.get(s.upper(), '#6B7280') for s in surgery_counts['Surgery']]
                        
                        fig = go.Figure(go.Bar(
                            x=surgery_counts['Count'],
                            y=surgery_counts['Surgery'],
                            orientation='h',
                            marker_color=colors,
                            text=surgery_counts['Count'],
                            textposition='outside'
                        ))
                        
                        fig.update_layout(
                            height=250,
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                            font=dict(color='#374151', size=12),
                            xaxis=dict(title='Count'),
                            yaxis=dict(title=''),
                            margin=dict(l=80, r=40, t=20, b=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # ===== ROW 6: INJURY TIMELINE =====
            st.subheader("Injury Timeline")
            
            if 'Date of Injury (MM/DD/YYYY)' in df.columns:
                df_timeline = df.copy()
                df_timeline['Injury Date'] = pd.to_datetime(df_timeline['Date of Injury (MM/DD/YYYY)'], errors='coerce')
                df_timeline = df_timeline[df_timeline['Injury Date'].notna()]
                df_timeline = df_timeline.sort_values('Injury Date')
                
                if len(df_timeline) > 0:
                    # Get individual scores for sizing
                    scores = df_timeline['Individual Score'].fillna(1)
                    
                    # Color by tissue type
                    tissue_colors = {
                        'MUSCLE': '#EF4444',
                        'CARTILAGE': '#3B82F6',
                        'TENDON': '#F59E0B',
                        'LIGAMENT': '#10B981',
                        'BONE': '#8B5CF6',
                        'NEURAL': '#EC4899'
                    }
                    colors = [tissue_colors.get(str(t).upper(), '#6B7280') for t in df_timeline['Tissue Type']]
                    
                    # Create labels
                    labels = df_timeline.apply(
                        lambda r: f"{r.get('BODY REGION', 'Unknown')} ({r.get('SIDE', '')})", axis=1
                    )
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df_timeline['Injury Date'],
                        y=scores,
                        mode='markers+text',
                        marker=dict(
                            size=scores * 5 + 10,
                            color=colors,
                            line=dict(width=2, color='white')
                        ),
                        text=labels,
                        textposition='top center',
                        textfont=dict(size=10, color='#374151'),
                        hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Score: %{y:.2f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        height=350,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=12),
                        xaxis=dict(title='Date'),
                        yaxis=dict(title='Individual Score'),
                        margin=dict(l=40, r=40, t=40, b=40),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # ===== ROW 7: RECENCY ANALYSIS =====
            st.subheader("Injury Recency Analysis")
            
            if 'Months Since' in df.columns:
                df_recency = df[df['Months Since'].notna() & (df['Months Since'] != '')].copy()
                df_recency['Months Since'] = pd.to_numeric(df_recency['Months Since'], errors='coerce')
                df_recency = df_recency[df_recency['Months Since'].notna()]
                
                if len(df_recency) > 0:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Recency histogram
                        fig = go.Figure()
                        
                        fig.add_trace(go.Histogram(
                            x=df_recency['Months Since'],
                            nbinsx=10,
                            marker_color='#3B82F6',
                            opacity=0.8
                        ))
                        
                        # Add zones
                        fig.add_vrect(x0=0, x1=12, fillcolor="red", opacity=0.1, annotation_text="Recent (<1yr)")
                        fig.add_vrect(x0=12, x1=36, fillcolor="yellow", opacity=0.1, annotation_text="Moderate (1-3yr)")
                        fig.add_vrect(x0=36, x1=max(df_recency['Months Since']), fillcolor="green", opacity=0.1, annotation_text="Old (3yr+)")
                        
                        fig.update_layout(
                            height=280,
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                            font=dict(color='#374151', size=12),
                            xaxis=dict(title='Months Since Injury'),
                            yaxis=dict(title='Count'),
                            margin=dict(l=40, r=40, t=20, b=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Recency summary
                        recent = len(df_recency[df_recency['Months Since'] < 12])
                        moderate = len(df_recency[(df_recency['Months Since'] >= 12) & (df_recency['Months Since'] < 36)])
                        old = len(df_recency[df_recency['Months Since'] >= 36])
                        
                        st.markdown(f"""
                        <div style="background-color: #F3F4F6; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                            <div style="color: #EF4444; font-size: 24px; font-weight: bold;">{recent}</div>
                            <div style="color: #6B7280; font-size: 12px;">Recent (&lt;1 year)</div>
                        </div>
                        <div style="background-color: #F3F4F6; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                            <div style="color: #F59E0B; font-size: 24px; font-weight: bold;">{moderate}</div>
                            <div style="color: #6B7280; font-size: 12px;">Moderate (1-3 years)</div>
                        </div>
                        <div style="background-color: #F3F4F6; padding: 15px; border-radius: 10px;">
                            <div style="color: #10B981; font-size: 24px; font-weight: bold;">{old}</div>
                            <div style="color: #6B7280; font-size: 12px;">Old (3+ years)</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ===== ROW 8: INJURY HISTORY TABLE =====
            st.subheader("Injury History Details")
            
            display_cols = ['LANDMARK', 'BODY REGION', 'SIDE', 'Tissue Type', 'Surgery', 
                           'Recovery Weeks', 'Date of Injury (MM/DD/YYYY)', 'Months Since', 'Individual Score']
            display_cols = [c for c in display_cols if c in df.columns]
            
            if display_cols:
                display_df = df[display_cols].copy()
                display_df = display_df[display_df['LANDMARK'].notna() & (display_df['LANDMARK'] != '')]
                
                if len(display_df) > 0:
                    st.dataframe(display_df.round(2), use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 5: INFO
# ============================================================================

elif page == "ℹ️ Info":
    st.title("Athlete Information")
    
    if 'info' not in data:
        st.warning("📋 Paste Info sheet URL in sidebar to view data")
    else:
        df = data['info'].copy()
        
        # Standardize name column
        if 'NAME' in df.columns and 'Athlete' not in df.columns:
            df = df.rename(columns={'NAME': 'Athlete'})
        
        if len(df) == 0:
            st.warning("No data available")
        else:
            # ===== ROW 1: SUMMARY CARDS =====
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total = df['Athlete'].nunique() if 'Athlete' in df.columns else len(df)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Athletes</div>
                    <div class="metric-value">{total}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                sports = df['SPORT'].nunique() if 'SPORT' in df.columns else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Sports</div>
                    <div class="metric-value">{sports}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_score = df['FINAL SCORE'].mean() if 'FINAL SCORE' in df.columns else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Final Score</div>
                    <div class="metric-value">{avg_score:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                if 'RISK TIER' in df.columns:
                    red_count = df['RISK TIER'].str.lower().str.contains('red').sum()
                else:
                    red_count = 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">High Risk (Red)</div>
                    <div class="metric-value" style="color: #EF4444;">{red_count}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ===== ROW 2: ATHLETE ROSTER =====
            st.subheader("Athlete Roster")
            
            # Create display dataframe
            display_cols = ['Athlete', 'AGE', 'SPORT', 'ASSESSMENT DATE', 'FINAL SCORE', 'RISK TIER']
            display_cols = [c for c in display_cols if c in df.columns]
            
            if display_cols:
                display_df = df[display_cols].copy()
                
                # Calculate age from DOB if AGE looks like a date
                if 'AGE' in display_df.columns:
                    try:
                        dob = pd.to_datetime(display_df['AGE'], errors='coerce')
                        if dob.notna().any():
                            today = pd.Timestamp.now()
                            display_df['AGE'] = ((today - dob).dt.days / 365.25).astype(int)
                    except:
                        pass
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # ===== ROW 3: CHARTS =====
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Athletes by Sport**")
                if 'SPORT' in df.columns:
                    sport_counts = df['SPORT'].value_counts().reset_index()
                    sport_counts.columns = ['Sport', 'Count']
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=sport_counts['Sport'],
                        values=sport_counts['Count'],
                        hole=0.4,
                        marker=dict(colors=['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']),
                        textinfo='value+percent',
                        textfont=dict(size=12, color='#374151'),
                        textposition='outside'
                    )])
                    
                    fig.update_layout(
                        height=300,
                        paper_bgcolor='white',
                        font=dict(color='#374151', size=12),
                        margin=dict(l=20, r=20, t=20, b=40),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(size=11, color='#374151'))
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Risk Tier Distribution**")
                if 'RISK TIER' in df.columns:
                    tier_counts = df['RISK TIER'].value_counts().reset_index()
                    tier_counts.columns = ['Tier', 'Count']
                    
                    color_map = {
                        'Green': '#10B981', 'green': '#10B981', 'GREEN': '#10B981',
                        'Yellow': '#F59E0B', 'yellow': '#F59E0B', 'YELLOW': '#F59E0B',
                        'Red': '#EF4444', 'red': '#EF4444', 'RED': '#EF4444',
                        'Blue': '#3B82F6', 'blue': '#3B82F6', 'BLUE': '#3B82F6'
                    }
                    colors = [color_map.get(t, '#6B7280') for t in tier_counts['Tier']]
                    
                    fig = go.Figure(go.Bar(
                        x=tier_counts['Tier'],
                        y=tier_counts['Count'],
                        marker_color=colors,
                        text=tier_counts['Count'],
                        textposition='outside',
                        textfont=dict(size=12, color='#374151')
                    ))
                    
                    fig.update_layout(
                        height=300,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=12),
                        xaxis=dict(title='', tickfont=dict(size=11, color='#4B5563'), showgrid=False),
                        yaxis=dict(title='Count', tickfont=dict(size=11, color='#4B5563'), gridcolor='#E5E7EB'),
                        margin=dict(l=40, r=40, t=20, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # ===== ROW 4: FINAL SCORE COMPARISON =====
            st.subheader("Final Score Comparison")
            
            if 'Athlete' in df.columns and 'FINAL SCORE' in df.columns:
                df_sorted = df.sort_values('FINAL SCORE', ascending=True)
                
                # Color by risk tier
                if 'RISK TIER' in df_sorted.columns:
                    color_map = {
                        'Green': '#10B981', 'green': '#10B981', 'GREEN': '#10B981',
                        'Yellow': '#F59E0B', 'yellow': '#F59E0B', 'YELLOW': '#F59E0B',
                        'Red': '#EF4444', 'red': '#EF4444', 'RED': '#EF4444',
                        'Blue': '#3B82F6', 'blue': '#3B82F6', 'BLUE': '#3B82F6'
                    }
                    colors = [color_map.get(t, '#6B7280') for t in df_sorted['RISK TIER']]
                else:
                    colors = '#3B82F6'
                
                fig = go.Figure(go.Bar(
                    x=df_sorted['FINAL SCORE'],
                    y=df_sorted['Athlete'],
                    orientation='h',
                    marker_color=colors,
                    text=df_sorted['FINAL SCORE'].round(1),
                    textposition='outside',
                    textfont=dict(size=11, color='#374151')
                ))
                
                # Add threshold lines
                fig.add_vline(x=85, line_dash="dash", line_color="#10B981", annotation_text="Green (85+)", annotation_font=dict(size=10, color='#10B981'))
                fig.add_vline(x=70, line_dash="dash", line_color="#F59E0B", annotation_text="Yellow (70+)", annotation_font=dict(size=10, color='#F59E0B'))
                
                fig.update_layout(
                    height=max(250, len(df_sorted) * 50),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='#374151', size=12),
                    xaxis=dict(range=[0, 100], title='Final Score', tickfont=dict(size=11, color='#4B5563'), gridcolor='#E5E7EB'),
                    yaxis=dict(title='', tickfont=dict(size=11, color='#4B5563')),
                    margin=dict(l=120, r=60, t=40, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # ===== ROW 5: INDIVIDUAL ATHLETE CARDS =====
            st.markdown("---")
            st.subheader("Athlete Cards")
            
            cols = st.columns(3)
            for idx, row in df.iterrows():
                col_idx = idx % 3
                
                name = row.get('Athlete', 'Unknown')
                sport = row.get('SPORT', 'N/A')
                score = row.get('FINAL SCORE', 0)
                tier = row.get('RISK TIER', 'N/A')
                
                # Calculate age
                age_val = row.get('AGE', 'N/A')
                try:
                    dob = pd.to_datetime(age_val, errors='coerce')
                    if pd.notna(dob):
                        age = int((pd.Timestamp.now() - dob).days / 365.25)
                    else:
                        age = age_val
                except:
                    age = age_val
                
                # Color based on tier
                tier_colors = {
                    'Green': ('#10B981', '#D1FAE5'),
                    'green': ('#10B981', '#D1FAE5'),
                    'GREEN': ('#10B981', '#D1FAE5'),
                    'Yellow': ('#F59E0B', '#FEF3C7'),
                    'yellow': ('#F59E0B', '#FEF3C7'),
                    'YELLOW': ('#F59E0B', '#FEF3C7'),
                    'Red': ('#EF4444', '#FEE2E2'),
                    'red': ('#EF4444', '#FEE2E2'),
                    'RED': ('#EF4444', '#FEE2E2'),
                    'Blue': ('#3B82F6', '#DBEAFE'),
                    'blue': ('#3B82F6', '#DBEAFE'),
                    'BLUE': ('#3B82F6', '#DBEAFE')
                }
                border_color, bg_color = tier_colors.get(tier, ('#6B7280', '#F3F4F6'))
                
                with cols[col_idx]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {bg_color}44, {bg_color}66); 
                                border: 2px solid {border_color}; 
                                border-radius: 12px; 
                                padding: 15px; 
                                margin: 10px 0;
                                text-align: center;">
                        <div style="font-size: 20px; font-weight: bold; color: #111827;">{name}</div>
                        <div style="font-size: 14px; color: #6B7280; margin: 5px 0;">{sport} | Age: {age}</div>
                        <div style="font-size: 32px; font-weight: bold; color: {border_color}; margin: 10px 0;">{score}</div>
                        <div style="background-color: {border_color}; color: white; padding: 5px 15px; 
                                    border-radius: 20px; display: inline-block; font-size: 12px; font-weight: bold;">
                            {tier.upper() if isinstance(tier, str) else tier}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("Athlete Operating System | Built with Ash")
