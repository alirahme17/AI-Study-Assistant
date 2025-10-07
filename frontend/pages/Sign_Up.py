import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.set_page_config(page_title="Teleco-Tutor - Sign Up", page_icon="✍️", layout="centered")

# --- User Authentication Setup ---
CONFIG_FILE = 'config.yaml'
with open(CONFIG_FILE) as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- Registration Form ---
st.title("Register New User")

try:
    if authenticator.register_user(location='main'):
        st.success('User registered successfully! Please go to the main page to login.')
        # Update the config file with the new user
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
except Exception as e:
    st.error(e)
