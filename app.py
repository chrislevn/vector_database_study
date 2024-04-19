import streamlit as st
from main import SimpleVectorDatabase
from helper import prepare_data

st.title('Simple Vector Database UI Demo')

st.write("Currently data is loaded from SICK2014 dataset. You can change the data source in helper.py")

input_sentences = prepare_data()
index_method = st.selectbox("Index Method", ["flat", "ivf", "hnsw", "pq", "sq"])
QUERY = st.text_input("Query", "Enter query here")
db = SimpleVectorDatabase(input_sentences, index_method=index_method)

if st.button("Run"):
    st.write("Running...")
    result = db.search(QUERY, top_k=1)
    st.write(result)