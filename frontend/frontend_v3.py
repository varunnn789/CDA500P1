import sys
from pathlib import Path
import os
import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium
import pytz

# Ensure proper shape file handling
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction

# Initialize session state
if "map_created" not in st.session_state:
    st.session_state.map_created = False

# Define EST timezone
est_tz = pytz.timezone("America/New_York")

# Get current time in EST
current_date = pd.Timestamp.now(tz="UTC").astimezone(est_tz)

st.title("New York Yellow Taxi Cab Demand Next Hour")
st.header(f'Current Time: {current_date.strftime("%Y-%m-%d %H:%M:%S %Z")}')

progress_bar = st.sidebar.progress(0)
N_STEPS = 4

# Step 1: Load NYC Taxi Zone Shapefile
with st.spinner("Loading NYC Taxi Zone Shapefile..."):
    shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"
    
    try:
        geo_df = gpd.read_file(shapefile_path)
        st.sidebar.write("✅ Shape file loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading shape file: {e}")
        st.stop()
    
    progress_bar.progress(1 / N_STEPS)

# Step 2: Load Inference Features
with st.spinner("Fetching batch of inference data..."):
    features = load_batch_of_features_from_store(current_date)

    if features.empty:
        st.sidebar.error("❌ No features available!")
        st.stop()

    # Convert timestamps to EST
    if "timestamp" in features.columns:
        features["timestamp"] = pd.to_datetime(features["timestamp"]).dt.tz_convert(est_tz)
    
    st.sidebar.write("✅ Features loaded")
    progress_bar.progress(2 / N_STEPS)

# Step 3: Fetch Predictions
with st.spinner("Fetching predictions..."):
    predictions = fetch_next_hour_predictions()

    if predictions.empty:
        st.sidebar.error("❌ No predictions available!")
        st.stop()

    # Convert timestamps to EST
    if "timestamp" in predictions.columns:
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"]).dt.tz_convert(est_tz)

    st.sidebar.write("✅ Predictions fetched")
    progress_bar.progress(3 / N_STEPS)

# Step 4: Plot Predictions on Map
with st.spinner("Generating taxi demand map..."):
    st.subheader("Taxi Ride Demand Prediction Map")

    def create_taxi_map(shapefile_path, prediction_data):
        """Creates an interactive choropleth map for NYC taxi demand."""
        nyc_zones = gpd.read_file(shapefile_path)

        # Merge with predictions
        nyc_zones = nyc_zones.merge(
            prediction_data[["pickup_location_id", "predicted_demand"]],
            left_on="LocationID",
            right_on="pickup_location_id",
            how="left",
        ).fillna(0)  # Replace NaN values with 0

        nyc_zones = nyc_zones.to_crs(epsg=4326)

        # Define color map
        colormap = LinearColormap(
            colors=["#FFEDA0", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C"],
            vmin=nyc_zones["predicted_demand"].min(),
            vmax=nyc_zones["predicted_demand"].max(),
        )
        colormap.add_to(m := folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron"))

        def style_function(feature):
            return {
                "fillColor": colormap(float(feature["properties"].get("predicted_demand", 0))),
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.7,
            }

        folium.GeoJson(
            nyc_zones.to_json(),
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(fields=["zone", "predicted_demand"], aliases=["Zone:", "Predicted Demand:"]),
        ).add_to(m)

        return m

    # Generate and display the map
    if not predictions.empty:
        st.session_state.map_obj = create_taxi_map(shapefile_path, predictions)
        st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])
    else:
        st.error("No predictions available to plot!")

    st.sidebar.write("✅ Taxi demand map generated")
    progress_bar.progress(4 / N_STEPS)

# Display Prediction Statistics
st.subheader("Prediction Statistics")
if not predictions.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Rides", f"{predictions['predicted_demand'].mean():.0f}")
    col2.metric("Maximum Rides", f"{predictions['predicted_demand'].max():.0f}")
    col3.metric("Minimum Rides", f"{predictions['predicted_demand'].min():.0f}")

    # Show sample of the predictions
    st.dataframe(predictions.sort_values("predicted_demand", ascending=False).head(10))

    # Plot top 10 locations
    for location_id in predictions.nlargest(10, "predicted_demand")["pickup_location_id"]:
        fig = plot_prediction(
            features=features[features["pickup_location_id"] == location_id],
            prediction=predictions[predictions["pickup_location_id"] == location_id],
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
else:
    st.error("❌ No prediction data available!")

st.sidebar.success("✅ All steps completed!")
