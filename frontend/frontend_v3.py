import sys
from pathlib import Path
import zipfile
import os
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium
import pytz

os.environ['SHAPE_RESTORE_SHX'] = 'YES'

from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# Initialize session state for the map
if "map_created" not in st.session_state:
    st.session_state.map_created = False

# Define EST timezone
est_tz = pytz.timezone("America/New_York")

# Get the current time in EST
current_date = pd.Timestamp.now(tz="UTC").astimezone(est_tz)

st.title("New York Yellow Taxi Cab Demand Next Hour")
st.header(f'{current_date.strftime("%Y-%m-%d %H:%M:%S %Z")}')

progress_bar = st.sidebar.header("Working Progress")
progress_bar = st.sidebar.progress(0)
N_STEPS = 4

# Step 1: Download and load NYC taxi zone shapefile
with st.spinner(text="Download shape file for taxi zones"):
    geo_df = gpd.read_file(DATA_DIR / "taxi_zones" / "taxi_zones.shp")
    st.sidebar.write("Shape file was downloaded")
    progress_bar.progress(1 / N_STEPS)

# Step 2: Fetch batch of inference data
with st.spinner(text="Fetching batch of inference data"):
    features = load_batch_of_features_from_store(current_date)
    
    # Convert timestamp column to EST (ensure the correct column name)
    if "timestamp" in features.columns:
        features["timestamp"] = pd.to_datetime(features["timestamp"]).dt.tz_convert(est_tz)
    
    st.sidebar.write("Inference features fetched from the store")
    progress_bar.progress(2 / N_STEPS)

# Step 3: Fetch predictions
with st.spinner(text="Fetching predictions"):
    predictions = fetch_next_hour_predictions()
    
    # Convert timestamp column to EST (ensure the correct column name)
    if "timestamp" in predictions.columns:
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"]).dt.tz_convert(est_tz)

    st.sidebar.write("Model was loaded from the registry")
    progress_bar.progress(3 / N_STEPS)

# Step 4: Create and display the taxi ride demand map
shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

with st.spinner(text="Plot predicted rides demand"):
    st.subheader("Taxi Ride Predictions Map")
    
    def create_taxi_map(shapefile_path, prediction_data):
        """
        Create an interactive choropleth map of NYC taxi zones with predicted rides.
        """
        nyc_zones = gpd.read_file(shapefile_path)

        # Merge with cleaned column names
        nyc_zones = nyc_zones.merge(
            prediction_data[["pickup_location_id", "predicted_demand"]],
            left_on="LocationID",
            right_on="pickup_location_id",
            how="left",
        )

        # Fill NaN values with 0 for predicted demand
        nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)

        # Convert to GeoJSON for Folium
        nyc_zones = nyc_zones.to_crs(epsg=4326)

        # Create map
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")

        # Create color map
        colormap = LinearColormap(
            colors=["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026"],
            vmin=nyc_zones["predicted_demand"].min(),
            vmax=nyc_zones["predicted_demand"].max(),
        )

        colormap.add_to(m)

        # Define style function
        def style_function(feature):
            predicted_demand = feature["properties"].get("predicted_demand", 0)
            return {
                "fillColor": colormap(float(predicted_demand)),
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.7,
            }

        # Convert GeoDataFrame to GeoJSON
        zones_json = nyc_zones.to_json()

        # Add the choropleth layer
        folium.GeoJson(
            zones_json,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=["zone", "predicted_demand"],
                aliases=["Zone:", "Predicted Demand:"],
                style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),
            ),
        ).add_to(m)

        # Store the map in session state
        st.session_state.map_obj = m
        st.session_state.map_created = True
        return m

    map_obj = create_taxi_map(shapefile_path, predictions)

    # Display the map
    if st.session_state.map_created:
        st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

    # Display data statistics
    st.subheader("Prediction Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Rides", f"{predictions['predicted_demand'].mean():.0f}")
    with col2:
        st.metric("Maximum Rides", f"{predictions['predicted_demand'].max():.0f}")
    with col3:
        st.metric("Minimum Rides", f"{predictions['predicted_demand'].min():.0f}")

    # Show sample of the data
    st.sidebar.write("Finished plotting taxi rides demand")
    progress_bar.progress(4 / N_STEPS)

# Display top 10 demand locations
st.dataframe(predictions.sort_values("predicted_demand", ascending=False).head(10))
top10 = predictions.sort_values("predicted_demand", ascending=False).head(10)["pickup_location_id"].to_list()

for location_id in top10:
    fig = plot_prediction(
        features=features[features["pickup_location_id"] == location_id],
        prediction=predictions[predictions["pickup_location_id"] == location_id],
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
