import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances

# ---------------- 1. Page Config ----------------
st.set_page_config(
    page_title="Château Analytics | Wine Profiler",
    page_icon="🍷",
    layout="wide"
)

# ---------------- 2. Luxury Styling ----------------
st.markdown("""
<style>
    /* Background and global font */
    .stApp {
        background: linear-gradient(rgba(15, 0, 0, 0.85), rgba(15, 0, 0, 0.85)), 
                    url("https://images.unsplash.com/photo-1506377247377-2a5b3b0ca7df?q=80&w=2070");
        background-size: cover;
        background-position: center;
        color: #f1faee;
    }

    /* Elegant Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }

    /* Luxury Header */
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 54px;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(to right, #d4af37, #f9e27d, #d4af37);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .subtitle {
        text-align: center;
        font-size: 20px;
        letter-spacing: 3px;
        color: #d4af37;
        text-transform: uppercase;
        margin-bottom: 40px;
    }

    /* Sophisticated Button */
    div.stButton > button {
        background: linear-gradient(45deg, #800000, #4a0000);
        color: #d4af37 !important;
        border: 1px solid #d4af37;
        border-radius: 30px;
        height: 4em;
        width: 100%;
        font-size: 20px;
        font-weight: bold;
        transition: 0.3s;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    div.stButton > button:hover {
        background: #d4af37;
        color: #1a0a0a !important;
        border: 1px solid #1a0a0a;
    }

    /* Results styling */
    .result-text {
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        font-size: 24px;
        background: rgba(212, 175, 55, 0.1);
        border: 1px solid #d4af37;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- 3. Load Data & Logic ----------------
@st.cache_data
def load_data_and_clusters():
    df = pd.read_csv("wine_clustering_data.csv")
    scaler = joblib.load("wine_scaler.pkl")
    X_scaled = scaler.transform(df)
    
    eps, min_samples = 2, 2
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_scaled)
    
    clusters = {}
    for label in set(cluster_labels):
        if label != -1:
            clusters[label] = X_scaled[cluster_labels == label]
    return scaler, clusters, eps

scaler, clusters, eps = load_data_and_clusters()

# ---------------- 4. Header Section ----------------
st.markdown('<div class="main-title">CHÂTEAU ANALYTICS</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Premium Wine Sommelier AI</div>', unsafe_allow_html=True)

# ---------------- 5. Input Interface ----------------
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='color:#d4af37; text-align:center;'>🍷 Chemical Profile Analysis</h3>", unsafe_allow_html=True)
    
    # Using 3 columns for better ergonomics
    c1, c2, c3 = st.columns(3)
    
    with c1:
        alcohol = st.slider("Alcohol", 11.0, 15.0, 13.0)
        malic_acid = st.slider("Malic Acid", 0.7, 5.8, 2.3)
        ash = st.slider("Ash", 1.3, 3.2, 2.3)
        ash_alcanity = st.slider("Ash Alcanity", 10.0, 30.0, 19.0)
    
    with c2:
        magnesium = st.slider("Magnesium", 70.0, 162.0, 100.0)
        total_phenols = st.slider("Total Phenols", 0.9, 3.8, 2.3)
        flavanoids = st.slider("Flavanoids", 0.3, 5.0, 2.0)
        nonflavanoid_phenols = st.slider("Nonflavanoid", 0.1, 0.7, 0.3)

    with c3:
        proanthocyanins = st.slider("Proanthocyanins", 0.4, 3.5, 1.5)
        color_intensity = st.slider("Color Intensity", 1.2, 13.0, 5.0)
        hue = st.slider("Hue", 0.4, 1.7, 1.0)
        proline = st.slider("Proline", 270.0, 1680.0, 750.0)

    st.markdown("<br>", unsafe_allow_html=True)
    identify = st.button("🔍 Begin Sommelier Assessment")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- 6. Prediction Logic ----------------
if identify:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Loading animation for "elegance"
    with st.spinner("⏳ Analyzing molecular structure and density..."):
        time.sleep(1.5)
        
        # Hardcoded fixed list as per the number of features the model expects
        input_data = np.array([[alcohol, malic_acid, ash, ash_alcanity, magnesium, 
                                total_phenols, flavanoids, nonflavanoid_phenols, 
                                proanthocyanins, color_intensity, hue, 2.5, proline]]) # added od280 placeholder

        scaled_input = scaler.transform(input_data)
        assigned_cluster = -1
        min_distance = float("inf")

        for label, cluster_points in clusters.items():
            distances = euclidean_distances(scaled_input, cluster_points)
            closest_distance = np.min(distances)
            if closest_distance < eps and closest_distance < min_distance:
                min_distance = closest_distance
                assigned_cluster = label

    # Display results inside a luxury card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if assigned_cluster == -1:
        st.markdown(f"""
            <div class="result-text" style="border-color: #ff4b4b; color: #ff4b4b;">
                ⚠️ <b>IDENTIFICATION: NOISE / OUTLIER</b><br>
                <small>This specimen does not follow standard cluster density patterns.</small>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-text">
                🏆 <b>AUTHENTICATION SUCCESSFUL</b><br>
                This wine belongs to <span style="color:#f9e27d;">VINTAGE CLUSTER {assigned_cluster}</span><br>
                <small>Distance to Core: {min_distance:.3f} units</small>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
