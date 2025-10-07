import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.set_page_config(page_title="Teleco-Tutor - Login", page_icon="📡", layout="centered")

# --- User Authentication ---
CONFIG_FILE = 'config.yaml'
with open(CONFIG_FILE) as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

st.title("Welcome to Teleco-Tutor 📡")
st.write("Please log in or visit the Sign Up page to create a new account.")

# --- Login Form ---
authenticator.login()

if st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.title(f"Welcome, {st.session_state['name']}!")
    st.success("You have successfully logged in!")
    st.info("Navigate to the Chat page from the sidebar to start asking questions.")

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')

elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password.')

