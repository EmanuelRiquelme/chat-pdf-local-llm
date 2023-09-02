import streamlit as st
import requests
import pandas as pd
import os

csv = st.file_uploader("Upload A CSV File", type=([".csv"]))

if csv:
    df = pd.read_csv(csv)
    csv = os.path.join(os.path.dirname(__file__),f'temp_{csv.name}')
    df.to_csv(f'{csv}', index=False)
    api_url = 'http://127.0.0.1:8000/get_file'
    params = {"csv_file": csv}
    response = requests.get(api_url, params=params)
    if response.status_code != 200:
        st.write(response.status_code)
    st.header('Ask a question')
    user_input = st.text_input("Enter some question:",placeholder = 'question')
    if user_input:
        api_url = 'http://127.0.0.1:8000/query'
        params = {"q": user_input}
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            api_data = response.json()
            st.write(api_data)
        else:
            st.write(response.status_code)
