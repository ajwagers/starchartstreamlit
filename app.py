import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import numpy as np

# Define column widths and headings
yalewidths = [4, 10, 11, 6, 6, 4, 1, 1, 1, 5, 2, 9, 2, 2, 4, 1, 2, 2, 2, 2, 2, 4, 1, 2, 2, 2, 6, 6, 5, 1, 1, 5, 1, 5, 1, 5, 1, 20, 1, 6, 6, 1, 5, 4, 4, 2, 3, 1, 4, 6, 4, 2, 1]
yaleheadings = ["HR", "Name", "DM", "HD", "SAO", "FK5", "IRflag", "r_IRflag", "Multiple", "ADS", "ADScomp", "VarID", "RAh1900", "RAm1900", "RAs1900", "DE-1900", "DEd1900", "DEm1900", "DEs1900", "RAh", "RAm", "RAs", "DE-", "DEd", "DEm", "DEs", "GLON", "GLAT", "Vmag", "n_Vmag", "u_Vmag", "B-V", "u_B-V", "U-B", "u_U-B", "R-I", "u_R-I", "SpType", "n_SpType", "pmRA", "pmDE", "n_Parallax", "Parallax", "RadVel", "n_RadVel", "l_RotVel", "RotVel", "u_RotVel", "Dmag", "Sep", "MultID", "MultCnt", "NoteFlag"]

@st.cache_data
def load_data():
    # Read the data file
    yalestarcatalog = pd.read_fwf("bsc5.dat", widths=yalewidths, header=None, names=yaleheadings)
    # Sort and filter the data
    yalestarcatalog = yalestarcatalog.sort_values(by='Vmag').head(9000)
    return yalestarcatalog

def calculate_star_positions(catalog, time, lat):
    # Calculate the Hour Angle for each star
    hour_angle = (time.hour + time.minute / 60 + time.second / 3600) - (catalog["RAh"] + catalog["RAm"] / 60 + catalog["RAs"] / 3600)
    hour_angle = hour_angle.apply(lambda x: x + 24 if x < 0 else x)
    hour_angle_degrees = hour_angle * 15

    # Convert Equatorial Coords (RA & DEC) to Horizon Coords (ALT & AZ)
    dec_degrees = catalog["DEd"] + catalog["DEm"] / 60 + catalog["DEs"] / 3600
    dec_degrees = dec_degrees.where(~catalog["DE-"].str.contains("-", na=False), -dec_degrees)

    # Convert to radians for faster computation
    lat_rad = math.pi * lat / 180
    dec_rad = dec_degrees * math.pi / 180
    hour_angle_rad = hour_angle_degrees * math.pi / 180

    # Calculate sine and cosine components
    sin_dec = np.sin(dec_rad)
    cos_dec = np.cos(dec_rad)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    cos_hour_angle = np.cos(hour_angle_rad)

    # Calculate altitude
    sinalt = sin_dec * sin_lat + cos_dec * cos_lat * cos_hour_angle
    alt = np.arcsin(sinalt) * 180 / math.pi

    # Calculate azimuth
    sin_hour_angle = np.sin(hour_angle_rad)
    cos_alt = np.cos(np.arcsin(sinalt))
    cosazm = (sin_dec - sinalt * sin_lat) / (cos_alt * cos_lat)
    cosazm = np.clip(cosazm, -1, 1)  # Handle any floating-point errors
    azm = np.arccos(cosazm) * 180 / math.pi
    azm = np.where(sin_hour_angle >= 0, azm, 360 - azm)

    return pd.Series(alt, index=catalog.index), pd.Series(azm, index=catalog.index)

def create_star_chart(catalog, alt, azm, lowest_alt, highest_mag):
    valid_stars = (alt >= lowest_alt) & (catalog['Vmag'] <= highest_mag)
    alt_valid = alt[valid_stars]
    azm_valid = azm[valid_stars]
    vmag_valid = catalog['Vmag'][valid_stars]
    
    x1 = np.cos(azm_valid * np.pi / 180) * np.tan(((90 - alt_valid) * np.pi / 180) / 2)
    y1 = np.sin(azm_valid * np.pi / 180) * np.tan(((90 - alt_valid) * np.pi / 180) / 2)

    # Calculate position for horizon (alt = 0 for all azimuths)
    azm_horizon = np.linspace(0, 360, 1000)
    x_horizon = np.cos(azm_horizon * np.pi / 180) * np.tan((90 * np.pi / 180) / 2)
    y_horizon = np.sin(azm_horizon * np.pi / 180) * np.tan((90 * np.pi / 180) / 2)

    # Set up the plot with black background
    plt.figure(figsize=(10, 10), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    # Draw the horizon line
    plt.plot(x_horizon, y_horizon, color='white', alpha=0.5, linewidth=1, linestyle='--')

    # Add cardinal directions
    directions = ['N', 'E', 'S', 'W']
    angles = [0, 90, 180, 270]
    for d, a in zip(directions, angles):
        x = np.cos(a * np.pi / 180) * np.tan((90 * np.pi / 180) / 2)
        y = np.sin(a * np.pi / 180) * np.tan((90 * np.pi / 180) / 2)
        plt.text(x, y, d, color='white', ha='center', va='center', alpha=0.7, fontsize=12)

    # Draw stars with size and brightness inversely proportional to magnitude
    sizes = 100 / (vmag_valid + 2)  # Squared for more dramatic size difference
    
    # Ensure alphas are between 0 and 1
    min_vmag = vmag_valid.min()
    max_vmag = vmag_valid.max()
    alphas = (max_vmag - vmag_valid + 1) / (max_vmag - min_vmag + 1)

    plt.scatter(x1, y1, s=sizes, alpha=alphas, edgecolor='none', color='white')

    # Flip y-axis to match astronomical convention (N up, E left)
    plt.gca().invert_xaxis()

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    return plt

# Streamlit app
st.title('Star Chart Maker')

# Sidebar for inputs
st.sidebar.header('Input Parameters')
date = st.sidebar.date_input('Date', value=datetime.now().date())
time = st.sidebar.time_input('Time', value=datetime.now().time())
timezone = st.sidebar.selectbox('Time Zone', pytz.common_timezones, index=pytz.common_timezones.index('US/Eastern'))
lat = st.sidebar.number_input('Latitude', value=39.0, min_value=-90.0, max_value=90.0, step=0.1)
lowest_alt = st.sidebar.number_input('Lowest Altitude', value=0, min_value=0, max_value=90, step=1)
highest_mag = st.sidebar.number_input('Highest Magnitude', value=6, min_value=1, max_value=9, step=1)

# Load data
yalestarcatalog = load_data()

# Calculate chart time in the selected time zone
tz = pytz.timezone(timezone)
chart_time = datetime.combine(date, time)
chart_time = tz.localize(chart_time).astimezone(pytz.UTC)

# Calculate star positions
alt, azm = calculate_star_positions(yalestarcatalog, chart_time, lat)

# Create and display the star chart
fig = create_star_chart(yalestarcatalog, alt, azm, lowest_alt, highest_mag)
st.pyplot(fig)

# Display instructions
st.sidebar.markdown("""
### Instructions:
1. Set the date, time, and time zone for your location.
2. Enter your latitude (e.g., 39.0 for Washington D.C.).
3. Adjust the lowest altitude and highest magnitude to control visibility.
4. The chart shows the sky as seen from your location, with the center being directly overhead.
""")