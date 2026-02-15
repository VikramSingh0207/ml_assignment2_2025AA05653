#Import Libraries
import streamlit as st

st.title('BITS Assignment-2 (2025AA05653) for Machine Learning')
st.write('Welcome to Assignment-2 of Machine Learning')

user_input = st.text_input("Enter a message:", "Hello")

st.write("Message: ", user_input)

