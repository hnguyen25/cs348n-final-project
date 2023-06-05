import streamlit as st



st.sidebar.header("CS 348N Final Project")
st.sidebar.subheader("Huy Nguyen")


st.sidebar.subheader("BuildingNet")
st.sidebar.text("Experiment #0: Baselines")
st.sidebar.text("Experiment #1: Generate random buildings of different classes")
st.sidebar.text("Experiment #2: Generate random buildings w/ semantic labels")
st.sidebar.text("Experiment #3: Generate random buildings w/ conditioned depth images")
st.sidebar.subheader("BuildingGAN")
st.sidebar.text("Experiment #1: Generate random volumetric designs")
st.sidebar.text("Experiment #2: Generate random volumetric designs w/ labels")
#house_type = st.dropdown(["Residential - House", "Religious - Church", "Religious - Mosque", "Religious - Temple", "Commercial - Office Building"])

# random samples from original datasest

# random samples from downsampled data
button = st.button("Generate random house")
