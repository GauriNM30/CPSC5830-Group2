# This will be the main streamlit UI file for the Capstone project.
# We will just call the app.py and some other stuff can be done here.
# Please add logging 

import streamlit as st
from old_app import Capstone

# Initialize the Capstone chatbot
capstone = Capstone()

# Streamlit UI setup
st.title("LLM Capstone Chatbot")

# Input field for user query
user_input = st.text_input("Enter your message:", placeholder="Type your message here...")

if st.button("Send"):
    if user_input:
        response = capstone.chat(user_input)
        st.text_area("Capstone Bot:", value=response, height=200)
    else:
        st.warning("Please enter a message.")
