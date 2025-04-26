import os
import streamlit as st
import json
import hashlib
from datetime import datetime

# Define the path for storing user chat data
DATA_DIR = "user_data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_user_id():
    """Get or create a user ID"""
    if 'user_id' not in st.session_state:
        # First try to load from query parameters using the new non-experimental API
        if 'user_id' in st.query_params:
            st.session_state.user_id = st.query_params['user_id']
        else:
            # If not logged in, show login form
            st.session_state.show_login = True
    
    return st.session_state.get('user_id', None)

def save_user_chat(user_id, messages):
    """Save the user's chat history to a JSON file"""
    if not user_id:
        return False
    
    user_file = os.path.join(DATA_DIR, f"{user_id}.json")
    chat_data = {
        "user_id": user_id,
        "last_updated": datetime.now().isoformat(),
        "messages": messages
    }
    
    try:
        with open(user_file, 'w') as f:
            json.dump(chat_data, f)
        return True
    except Exception as e:
        st.error(f"Failed to save chat history: {str(e)}")
        return False

def load_user_chat(user_id):
    """Load a user's chat history from their JSON file"""
    if not user_id:
        return []
    
    user_file = os.path.join(DATA_DIR, f"{user_id}.json")
    
    try:
        if os.path.exists(user_file):
            with open(user_file, 'r') as f:
                chat_data = json.load(f)
                return chat_data.get("messages", [])
        else:
            return []
    except Exception as e:
        st.error(f"Failed to load chat history: {str(e)}")
        return []

def handle_login(username, password):
    """Simple login handler - in production use a more secure method"""
    # Create a simple hash of username+password as the user ID
    # In production, use a proper authentication system
    if not username or not password:
        return None
    
    user_id = hashlib.md5(f"{username}:{password}".encode()).hexdigest()
    st.session_state.user_id = user_id
    st.session_state.username = username
    st.session_state.show_login = False
    
    # Set the user_id in URL parameters for bookmarking - using new API
    st.query_params['user_id'] = user_id
    
    return user_id

def show_login_form():
    """Render login form and handle submissions"""
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    st.subheader("Login or Sign Up")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login / Sign Up")
        
        if submit_button:
            if username and password:
                user_id = handle_login(username, password)
                if user_id:
                    # Return True to indicate successful login
                    return True, user_id
            else:
                st.error("Please enter both username and password")
    
    st.markdown("</div>", unsafe_allow_html=True)
    return False, None

def show_user_header(primary_color):
    """Display user header with logout functionality using pure Streamlit"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
            <div style='padding: 5px 15px;'>
                <span style='font-weight:bold;'>👤 {st.session_state.get('username', 'User')}</span>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("Log Out", key="logout_button", type="primary", use_container_width=True):
            # Clear session state
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.show_login = True
            
            # Clear URL parameters
            st.query_params.clear()
            
            # Force a rerun to refresh the page
            st.rerun()