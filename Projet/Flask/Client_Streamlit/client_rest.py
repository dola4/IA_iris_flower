import streamlit as st
import json 
import requests
import pandas as pd 
import numpy as np

url = "127.0.0.1:8009/model"

st.write("""
        # simple Iris flower Prediction App
        # This App predict Iris flower types
        """)
st.sidebar.header("User Input parameters")

