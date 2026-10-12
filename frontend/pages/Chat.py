import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import requests
import os
import time
from typing import List, Generator

# --- Page Config ---
st.set_page_config(page_title="Teleco-Tutor - Chat", page_icon="💬", layout="centered")

# --- Constants ---
API_URL = os.getenv("TELECO_TUTOR_API_URL", "http://127.0.0.1:8000")
CONFIG_FILE = 'config.yaml'

# --- Authentication Check ---
# This is the most important part of a multi-page app with authentication
with open(CONFIG_FILE) as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

if not st.session_state.get("authentication_status"):
    st.warning("Please log in from the main page to access the chat.")
    st.stop() # Stop further execution of the page

# --- Logged-in User Info ---
name = st.session_state.get("name")
username = st.session_state.get("username")
authenticator.logout('Logout', 'sidebar')
st.sidebar.title(f"Welcome, {name}!")


# --- Backend API Functions (from your original file) ---
@st.cache_data(ttl=3600)
def get_courses(api_url: str, username: str) -> List[str]:
    try:
        resp = requests.get(f"{api_url}/api/courses/{username}")
        if resp.status_code == 200:
            return resp.json().get("courses", [])
        return []
    except requests.exceptions.RequestException:
        return ["Error: Could not fetch courses"]

def stream_query_backend(question: str, api_url: str, username: str, course: str) -> Generator[str, None, None]:
    payload = {"username": username, "question": question, "course": course}
    try:
        with requests.post(f"{api_url}/api/query", json=payload, stream=True, timeout=60) as resp:
            if resp.status_code == 200:
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk.decode('utf-8')
            else:
                error_message = f"**Error:** Backend responded with status {resp.status_code}. {resp.text[:300]}"
                yield error_message
    except requests.exceptions.RequestException as e:
        yield f"**Error:** Failed to connect to backend: {e}"


# --- Main Application Logic ---
POLL_SECONDS = 5
MAX_WAIT_SECONDS = 60 * 60  # stop waiting after an hour and tell the user


def fetch_status() -> dict:
    """Returns {"status": not_configured|processing|ready|error|unreachable, "message": str}."""
    try:
        resp = requests.get(f"{API_URL}/api/status/{username}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return {"status": "error", "message": f"Backend returned {resp.status_code}: {resp.text[:200]}"}
    except requests.exceptions.RequestException:
        return {"status": "unreachable", "message": "Could not connect to the backend. Is the server running?"}


def wait_for_processing():
    """Poll until the build finishes (ready or error), showing live progress."""
    progress = st.empty()
    started = time.time()
    with st.spinner("Building your knowledge base... This can take several minutes."):
        while time.time() - started < MAX_WAIT_SECONDS:
            current = fetch_status()
            if current["status"] != "processing":
                st.rerun()
            progress.info(current.get("message") or "Processing...")
            time.sleep(POLL_SECONDS)
    progress.warning("This is taking longer than expected. Refresh the page to check again.")


status = fetch_status()
is_configured = status["status"] == "ready"

if status["status"] == "unreachable":
    st.error(status["message"])
    st.stop()

if status["status"] == "processing":
    st.header("First-Time Setup")
    wait_for_processing()
    st.stop()

if not is_configured:
    # --- CONFIGURATION UI ---
    st.header("First-Time Setup")
    if status["status"] == "error":
        st.error(f"The last setup attempt failed: {status.get('message')}")
    st.info("Please provide your Google Drive link and Gemini API key to build your personal knowledge base.")
    drive_url = st.text_input("Google Drive Folder URL", help="The URL of the main folder containing all your course materials.")
    gemini_key = st.text_input("Gemini API Key", type="password", help="Get your key from Google AI Studio.")
    if st.button("Build My Knowledge Base", use_container_width=True):
        if not drive_url or not gemini_key:
            st.error("Please provide both a Drive URL and an API key.")
        else:
            payload = {"username": username, "drive_folder_url": drive_url, "gemini_api_key": gemini_key}
            try:
                resp = requests.post(f"{API_URL}/api/configure", json=payload, timeout=30)
                if resp.status_code == 200:
                    st.success("Configuration sent! Your documents are being processed.")
                    wait_for_processing()
                elif resp.status_code == 409:
                    wait_for_processing()
                else:
                    detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    st.error(f"Error from backend: {detail}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend: {e}")
else:
    # --- MAIN CHAT UI ---
    st.title(f"Teleco-Tutor for {name}")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Select a course and ask a question."}]
    
    courses = get_courses(API_URL, username)
    tabs = st.tabs(["Course", "Chat"])
    with tabs[0]:
        if not courses:
            st.error("Could not load your courses from the backend.")
        else:
            st.session_state.course = st.selectbox("Course", courses, help="Select the course to search within.")
        st.info(f"Current course: **{st.session_state.get('course', 'None')}**")
    with tabs[1]:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
        if user_msg := st.chat_input(f"Ask a question about {st.session_state.get('course', '...')}…"):
            st.session_state.messages.append({"role": "user", "content": user_msg})
            st.chat_message("user").write(user_msg)
            with st.chat_message("assistant"):
                response_generator = stream_query_backend(user_msg, API_URL, username, st.session_state.get('course'))
                full_response = st.write_stream(response_generator)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
