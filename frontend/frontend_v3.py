import sys
from pathlib import Path
import zipfile
import os
import folium
import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium
from datetime import datetime
import pytz
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction

# Function to convert UTC time to EST
def convert_to_est(utc_time):
    est_tz = pytz.timezone('US/Eastern')
    if utc_time.tzinfo is None:
        utc_time = pytz.utc.localize(utc_time)
    return utc_time.astimezone(est_tz)

if "map_created" not in st.session_state:
    st.session_state.map_created = False

def create_taxi_map(shapefile_path, prediction_data):
    nyc_zones = gpd.read_file(shapefile_path)
    nyc_zones = nyc_zones.merge(
        prediction_data[["pickup_location_id", "predicted_demand"]],
        left_on="LocationID",
        right_on="pickup_location_id",
        how="left"
    )
    nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)
    nyc_zones = nyc_zones.to_crs(epsg=4326)

    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")
    colormap = LinearColormap(
        colors=["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026"],
        vmin=nyc_zones["predicted_demand"].min(),
        vmax=nyc_zones["predicted_demand"].max()
    )
    colormap.add_to(m)

    def style_function(feature):
        predicted_demand = feature["properties"].get("predicted_demand", 0)
        return {
            "fillColor": colormap(float(predicted_demand)),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.7,
        }

    zones_json = nyc_zones.to_json()
    folium.GeoJson(
        zones_json,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["zone", "predicted_demand"],
            aliases=["Zone:", "Predicted Demand:"],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
        )
    ).add_to(m)

    st.session_state.map_obj = m
    st.session_state.map_created = True
    return m

def load_shape_data_file(data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", log=True):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "taxi_zones.zip"
    extract_path = data_dir / "taxi_zones"
    shapefile_path = extract_path / "taxi_zones.shp"

    if not zip_path.exists():
        if log:
            print(f"Downloading file from {url}...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(response.content)
        if log:
            print(f"File downloaded and saved to {zip_path}")
    else:
        if log:
            print(f"File already exists at {zip_path}, skipping download.")

    if not shapefile_path.exists():
        if log:
            print(f"Extracting files to {extract_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        if log:
            print(f"Files extracted to {extract_path}")
    else:
        if log:
            print(f"Shapefile already exists at {shapefile_path}, skipping extraction.")

    if log:
        print(f"Loading shapefile from {shapefile_path}...")
    gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
    if log:
        print("Shapefile successfully loaded.")
    return gdf

# Get current time in EST for display
current_date = pd.Timestamp.now(tz="Etc/UTC")
current_date_est = convert_to_est(current_date)
st.title(f"New York Yellow Taxi Cab Demand Next Hour")
st.header(f'{current_date_est.strftime("%Y-%m-%d %H:%M:%S %Z")}')

progress_bar = st.sidebar.header("Working Progress")
progress_bar = st.sidebar.progress(0)
N_STEPS = 4

with st.spinner(text="Download shape file for taxi zones"):
    geo_df = load_shape_data_file(DATA_DIR)
    st.sidebar.write("Shape file was downloaded")
    progress_bar.progress(1 / N_STEPS)

# Using current time in UTC for fetching data
with st.spinner(text="Fetching batch of inference data"):
    try:
        features = load_batch_of_features_from_store(current_date)
        if features is None or features.empty:
            st.warning("No features data available.")
            logging.warning("No features data available from load_batch_of_features_from_store.")
        else:
            if 'pickup_hour' in features.columns:
                features['pickup_hour'] = features['pickup_hour'].apply(lambda x: convert_to_est(pd.to_datetime(x)) if pd.notnull(x) else None)
            st.sidebar.write("Inference features fetched from the store")
            logging.info("Inference features fetched successfully.")
    except Exception as e:
        st.error(f"Error fetching features: {e}")
        features = pd.DataFrame()  # Assign an empty DataFrame in case of error
        logging.exception("Error fetching features from the store.")
    progress_bar.progress(2 / N_STEPS)

with st.spinner(text="Fetching predictions"):
    try:
        predictions = fetch_next_hour_predictions()
        if predictions is None or predictions.empty:
            st.warning("No prediction data available.")
            logging.warning("No prediction data available from fetch_next_hour_predictions.")
        else:
            if 'pickup_hour' in predictions.columns:
                predictions['pickup_hour'] = predictions['pickup_hour'].apply(lambda x: convert_to_est(pd.to_datetime(x)) if pd.notnull(x) else None)
            st.sidebar.write("Model was loaded from the registry")
            logging.info("Predictions fetched successfully.")
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        predictions = pd.DataFrame()  # Assign an empty DataFrame in case of error
        logging.exception("Error fetching predictions.")
    progress_bar.progress(3 / N_STEPS)

shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

with st.spinner(text="Plot predicted rides demand"):
    if predictions is None or predictions.empty:
        st.warning("No prediction data available to plot.")
    else:
        st.subheader("Taxi Ride Predictions Map")
        map_obj = create_taxi_map(shapefile_path, predictions)

        if st.session_state.map_created:
            st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

        st.subheader("Prediction Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Rides", f"{predictions['predicted_demand'].mean():.0f}")
        with col2:
            st.metric("Maximum Rides", f"{predictions['predicted_demand'].max():.0f}")
        with col3:
            st.metric("Minimum Rides", f"{predictions['predicted_demand'].min():.0f}")

        st.sidebar.write("Finished plotting taxi rides demand")
        progress_bar.progress(4 / N_STEPS)

        st.dataframe(predictions.sort_values("predicted_demand", ascending=False).head(10))
        top10 = (
            predictions.sort_values("predicted_demand", ascending=False)
            .head(10)["pickup_location_id"]
            .to_list()
        )
        for location_id in top10:
            # Check if location_id exists in both features and predictions
            if (location_id in features["pickup_location_id"].values) and \
               (location_id in predictions["pickup_location_id"].values):
                fig = plot_prediction(
                    features=features[features["pickup_location_id"] == location_id],
                    prediction=predictions[predictions["pickup_location_id"] == location_id],
                )
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            else:
                st.warning(f"No data available for location ID: {location_id}")
