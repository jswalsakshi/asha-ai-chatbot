import os
import sys
import streamlit as st
from streamlit_chat import message
import random
import json
import hashlib
from datetime import datetime
import importlib.util
import base64
from fpdf import FPDF
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# Define the path for storing user data
DATA_DIR = "user_data"
os.makedirs(DATA_DIR, exist_ok=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import resume builder functionality
from frontend.resume_builder import generate_resume_content, create_pdf_resume, get_pdf_download_link

# Import jobs recommendation service
jobs_reco_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "services", "jobs_reco.py")
if os.path.exists(jobs_reco_path):
    spec = importlib.util.spec_from_file_location("jobs_reco", jobs_reco_path)
    jobs_reco = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jobs_reco)
    # Import the specific functions
    get_recommended_jobs = jobs_reco.get_recommended_jobs
    format_job_listings = jobs_reco.format_job_listings
    load_jobs_from_csv = jobs_reco.load_jobs_from_csv
    search_jobs = jobs_reco.search_jobs
    # st.sidebar.success("✅ Jobs recommendation service loaded!")
else:
    st.sidebar.error(f"⚠️ Could not find jobs_reco.py at {jobs_reco_path}")
    # Define fallback functions
    def get_recommended_jobs(_, job_database=None, num_jobs=3):
        return []
    def format_job_listings(_):
        return "Job listings service is not available."
    def load_jobs_from_csv():
        return {}
    def search_jobs(_, job_database=None, max_results=5):
        return []

# Create user identification function
def get_user_id():
    """Get or create a user ID"""
    if 'user_id' not in st.session_state:
        # First try to load from query parameters
        query_params = st.query_params
        if 'user_id' in query_params:
            st.session_state.user_id = query_params['user_id']
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
    
    # Set the user_id in URL parameters for bookmarking
    st.query_params.user_id = user_id
    
    return user_id

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Try to find the rag_service.py file
rag_service_path = None
default_rag_path = os.path.join(current_dir, "..", "app", "services", "rag_service.py")
if os.path.isfile(default_rag_path):
    rag_service_path = os.path.abspath(default_rag_path)

for root, dirs, files in os.walk(current_dir):
    if "rag_service.py" in files:
        rag_service_path = os.path.join(root, "rag_service.py")
        break


@st.cache_resource(show_spinner=False)
def get_rag_service():
    """Load RAG service resources - only called once and cached"""
    try:
        if not rag_service_path:
            raise ImportError("Could not find rag_service.py file")
            
        # Import directly from file path
        spec = importlib.util.spec_from_file_location("rag_service", rag_service_path)
        rag_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rag_module)
        
        return {
            "available": True,
            "search": rag_module.semantic_search,
            "generate": rag_module.generate_response_huggingface,
            "error": None
        }
    except Exception as e:
        return {
            "available": False,
            "search": None, 
            "generate": None,
            "error": str(e)
        }

# Test RAG service and show diagnostic info at startup
try:
    rag_test = get_rag_service()
    if not rag_test["available"]:
        st.sidebar.error(f"⚠️ RAG service not available: {rag_test['error']}")
    # else:
        # st.sidebar.success("✅ RAG service connected successfully!")
except Exception as e:
    st.sidebar.error(f"⚠️ Error testing RAG service: {str(e)}")

# Try to find the llm_service.py file
llm_service_path = None
default_llm_path = os.path.join(current_dir, "..", "app", "services", "llm_service.py")
if os.path.isfile(default_llm_path):
    llm_service_path = os.path.abspath(default_llm_path)

for root, dirs, files in os.walk(current_dir):
    if "llm_service.py" in files:
        llm_service_path = os.path.join(root, "llm_service.py")
        break

@st.cache_resource(show_spinner=False)
def get_llm_service():
    """Load LLM service resources - only called once and cached"""
    try:
        if not llm_service_path:
            raise ImportError("Could not find llm_service.py file")
            
        # Import directly from file path
        import importlib.util
        spec = importlib.util.spec_from_file_location("llm_service", llm_service_path)
        llm_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(llm_module)
        
        return {
            "available": True,
            "generate_career_advice": llm_module.generate_career_advice,
            "generate_course_recommendations": llm_module.generate_course_recommendations,
            "generate_interview_resources": llm_module.generate_interview_resources,
            "is_career_question": llm_module.is_career_question,
            "error": None
        }
    except Exception as e:
        return {
            "available": False,
            "generate_career_advice": None,
            "is_career_question": None,
            "generate_course_recommendations": None,
            "generate_interview_resources": None,
            "error": str(e)
        }

# Test LLM service and show diagnostic info at startup
try:
    llm_test = get_llm_service()
    if not llm_test["available"]:
        st.sidebar.error(f"⚠️ LLM service not available: {llm_test['error']}")
    # else:
    #     st.sidebar.success("✅ LLM service connected successfully!")
except Exception as e:
    st.sidebar.error(f"⚠️ Error testing LLM service: {str(e)}")

@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_job_database():
    """Get job database with caching"""
    jobs = load_jobs_from_csv()
    # st.sidebar.success(f"✅ Loaded {sum(len(jobs[cat]) for cat in jobs)} jobs")
    return jobs

# Get the job database once
JOB_DATABASE = get_job_database()

# Set color theme
PRIMARY_COLOR = "#89506B"  # Changed to requested deep mauve/purple
SECONDARY_COLOR = "#B987A9"  # Light version of the main color
ACCENT_COLOR = "#694057"  # Darker version of the main color
BACKGROUND_COLOR = "#F5EDF2"  # Very light version of the main color
TEXT_COLOR = "#263238"  # Kept dark text
CHATBOT_BG_COLOR = "#A9C97D"  # Light green for chatbot messages

# Custom CSS
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {BACKGROUND_COLOR};
        }}
        .stChatFloatingInputContainer {{
            background-color: white !important;
            border: 2px solid {PRIMARY_COLOR} !important;
        }}
        .context-tabs {{
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        .context-tab {{
            padding: 8px 15px;
            background-color: {BACKGROUND_COLOR};
            border: 1px solid {PRIMARY_COLOR};
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }}
        .context-tab:hover {{
            background-color: {SECONDARY_COLOR};
        }}
        .context-tab.active {{
            background-color: {PRIMARY_COLOR};
            color: white;
        }}
        .login-container {{
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .user-header {{
            background-color: white;
            padding: 10px 15px;
            border-radius: 30px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .download-button {{
            display: inline-block;
            padding: 10px 20px;
            background-color: {PRIMARY_COLOR};
            color: white !important;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            margin-top: 10px;
            text-align: center;
        }}
        .download-button:hover {{
            background-color: {ACCENT_COLOR};
        }}
    </style>
""", unsafe_allow_html=True)

# Context options
CONTEXT_OPTIONS = {
    "general": {
        "name": "General Help",
        "icon": "💬",
        "welcome": "Hello! I'm Asha, your career companion. How can I help you today?",
        "color": PRIMARY_COLOR
    },
    "jobs": {
        "name": "Job Search",
        "icon": "💼",
        "welcome": "I'll help you find your ideal career opportunity! I've selected some job listings that match your profile. What specific roles or industries interest you?",
        "color": "#5C6BC0"
    },
    "resume": {
        "name": "Resume Help",
        "icon": "📝",
        "welcome": "Welcome to the Resume Builder! I can help you create a professional resume by asking you a series of questions. Just say 'build resume' or 'create resume' to get started.",
        "color": "#26A69A"
    },
    "interview": {
        "name": "Interview Prep",
        "icon": "🎤",
        "welcome": "Let's prepare for your interviews! What type of questions do you need help with?",
        "color": "#AB47BC"
    },
    "mentorship": {
        "name": "Mentorship",
        "icon": "👩‍🏫",
        "welcome": "Connecting you with mentors in your field. What area are you interested in?",
        "color": "#FFA000"
    }
}

# Resume questions sequence
resume_questions = [
    "What is your full name?",
    "What is your email address?",
    "What is your phone number?",
    "What is your location (City, State/Country)?",
    "Please write a brief professional summary (2-3 sentences about your background and career goals).",
    "Please share your education details (Degree, Institution, Year).",
    "Please describe your work experience (include positions, companies, dates, and key responsibilities).",
    "What are your key skills? (comma-separated list)"
]

resume_fields = [
    "name",
    "email",
    "phone",
    "location",
    "summary",
    "education",
    "experience",
    "skills"
]

SAMPLE_JOBS = [
    "Senior Product Designer at TechSolutions (Bangalore)",
    "Marketing Manager at BrandMakers (Remote)",
    "Software Engineer at WomenFirst Tech (Hybrid)"
]


@st.cache_data(ttl=300, max_entries=100, show_spinner=False)
def get_ai_response(query, context_name):
    """Get AI response with caching for identical queries and context"""
    # For general context, use RAG
    if context_name == "general":
        rag = get_rag_service()
        if rag["available"]:
            try:
                # Get relevant context - use explicit parameter names
                context_docs = rag["search"](query=query)
                context_text = "\n".join(context_docs)
                
                # Generate response - use explicit parameter names
                return rag["generate"](query=query, context=context_text)
            except Exception as e:
                st.error(f"RAG Error: {str(e)}")  # Add error logging
                pass  # Fallback to rule-based responses

    if context_name == "jobs":
        # Use RAG for job recommendations if available
        rag = get_rag_service()
        if rag["available"]:
            try:
                job_contexts = rag["search"](query="top jobs for career advancement")
                job_details = "\n".join(job_contexts[:2])  # Use top 2 contexts
                return f"I found 3 relevant jobs matching your profile based on {job_details}. Would you like me to share the details?"
            except:
                pass
        # Fallback to default response
        return "I found 3 relevant jobs matching your profile. Would you like me to share the details?"
    elif context_name == "resume":
        return "I can help you create a professional resume! Just say 'create resume' to get started, or ask me specific questions about resume writing."
    elif context_name == "interview":
        return "Let's practice interview questions! Are you preparing for technical or behavioral interviews?"
    elif context_name == "mentorship":
        return "We have mentors available in tech, business, and creative fields. What area interests you?"
    else:
        return "I'm happy to help with your career journey! What specific questions do you have today?"

# Initialize session state
if 'show_login' not in st.session_state:
    st.session_state.show_login = False

if 'job_preferences' not in st.session_state:
    st.session_state.job_preferences = None

if 'context' not in st.session_state:
    st.session_state.context = "general"

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize resume builder session states
if "resume_mode" not in st.session_state:
    st.session_state.resume_mode = False

if "current_question" not in st.session_state:
    st.session_state.current_question = 0

if "user_info" not in st.session_state:
    st.session_state.user_info = {
        "name": "",
        "email": "",
        "phone": "",
        "location": "",
        "summary": "",
        "education": "",
        "experience": "",
        "skills": ""
    }

if "resume_data" not in st.session_state:
    st.session_state.resume_data = None

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# Ensure Gemini API key is available
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key:
    st.session_state.gemini_api_key = gemini_api_key
    genai.configure(api_key=gemini_api_key)

# Header with logo
st.markdown(f"""
    <div style='background-color:{PRIMARY_COLOR};padding:15px;border-radius:10px;margin-bottom:20px;text-align:center;'>
        <h1 style='color:white;margin:0;'>Asha Chatbot</h1>
        <p style='color:white;margin:0;'>Your personal career assistant</p>
    </div>
""", unsafe_allow_html=True)

# Check if user is logged in or needs to login
user_id = get_user_id()

# Login form if needed
if 'show_login' in st.session_state and st.session_state.show_login:
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
                    # Load user chat history
                    loaded_messages = load_user_chat(user_id)
                    if loaded_messages:
                        st.session_state.messages = loaded_messages
                    else:
                        # New user, show welcome message
                        st.session_state.messages = [{
                            "role": "assistant",
                            "content": CONTEXT_OPTIONS[st.session_state.context]["welcome"]
                        }]
                    st.rerun()
            else:
                st.error("Please enter both username and password")
    
    st.markdown("</div>", unsafe_allow_html=True)
elif user_id:  # Only show the chat interface if logged in
    # Show user info if logged in
    header_col1, header_col2 = st.columns([5, 1])
    with header_col1:
        st.markdown(f"""
            <div style='padding: 10px 15px;'>
                <span style='font-weight:bold;'>👤 {st.session_state.get('username', 'User')}</span>
            </div>
        """, unsafe_allow_html=True)

    with header_col2:
        # Red power icon button for logout
        if st.button("⏻", key="logout_button", help="Logout", 
                    type="secondary", use_container_width=True):
            # Clear session state
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.show_login = True
            st.session_state.messages = []
            # Reset URL parameters
            st.query_params.clear()
            st.rerun()
    
    # Context tabs inside chat area
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("💬 General", key="tab_general_1", use_container_width=True):
            st.session_state.context = "general"
            st.session_state.messages.append({
                "role": "assistant", 
                "content": CONTEXT_OPTIONS["general"]["welcome"]
            })
            # Save updated chat
            save_user_chat(user_id, st.session_state.messages)
            st.rerun()
    
    with col2:
        if st.button("💼 Jobs", key="tab_jobs_1", use_container_width=True):
            st.session_state.context = "jobs"
        
            # Get top 3 jobs immediately when switching to jobs tab
            recommended_jobs = get_recommended_jobs(st.session_state.messages, JOB_DATABASE)
            job_response = format_job_listings(recommended_jobs)
        
            # Store with metadata for context
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"Here are the top 3 jobs matching your profile:\n\n{job_response}\n\nWhat type of jobs are you interested in? Or would you like help preparing your application?",
                "metadata": {
                    "type": "job_listings",
                    "jobs": recommended_jobs
                }
            })
            save_user_chat(user_id, st.session_state.messages)
            st.rerun()
    
    with col3:
        if st.button("📝 Resume", key="tab_resume_1", use_container_width=True):
            st.session_state.context = "resume"
            # Reset resume builder state when switching to resume tab
            st.session_state.resume_mode = False
            st.session_state.current_question = 0
            st.session_state.user_info = {
                "name": "", "email": "", "phone": "", "location": "",
                "summary": "", "education": "", "experience": "", "skills": ""
            }
            st.session_state.resume_data = None
            st.session_state.pdf_path = None
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": CONTEXT_OPTIONS["resume"]["welcome"]
            })
            save_user_chat(user_id, st.session_state.messages)
            st.rerun()
    
    with col4:
        if st.button("🎤 Interview", key="tab_interview_1", use_container_width=True):
            st.session_state.context = "interview"
            st.session_state.messages.append({
                "role": "assistant",
                "content": CONTEXT_OPTIONS["interview"]["welcome"]
            })
            save_user_chat(user_id, st.session_state.messages)
            st.rerun()
    
    with col5:
        if st.button("👩‍🏫 Mentorship", key="tab_mentorship_1", use_container_width=True):
            st.session_state.context = "mentorship"
            st.session_state.messages.append({
                "role": "assistant",
                "content": CONTEXT_OPTIONS["mentorship"]["welcome"]
            })
            save_user_chat(user_id, st.session_state.messages)  
            st.rerun()

    # Add chat history actions in the sidebar
    st.sidebar.subheader("Chat History")
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": CONTEXT_OPTIONS[st.session_state.context]["welcome"]
        }]
        save_user_chat(user_id, st.session_state.messages)
        st.rerun()
    
    # if st.sidebar.button("Export Chat as JSON"):
    #     chat_json = json.dumps(st.session_state.messages, indent=2)
    #     st.sidebar.download_button(
    #         label="Download JSON",
    #         data=chat_json,
    #         file_name=f"asha_chat_{user_id[:8]}_{datetime.now().strftime('%Y%m%d')}.json",
    #         mime="application/json"
    #     )

    if st.sidebar.button("Refresh Job Database"):
        # Clear the cache to force reload
        get_job_database.clear()
        # Reload
        JOB_DATABASE = get_job_database()
        st.sidebar.success("✅ Job database refreshed!")
        
    # Display messages
    for i, msg in enumerate(st.session_state.messages):
        is_user = msg["role"] == "user"
        
        # Define avatar URLs
        bot_avatar_url = "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"  
        user_avatar_url = "https://cdn-icons-png.flaticon.com/512/4140/4140047.png"  
        
        # Use Streamlit's native chat_message component
        with st.chat_message(msg["role"], avatar=user_avatar_url if is_user else bot_avatar_url):
            # Always use st.markdown with unsafe_allow_html=True for assistant messages
            # that might contain HTML (especially download links)
            if not is_user:
                st.markdown(msg["content"], unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])
   
    # Chat input - ONLY ONE INSTANCE placed here inside the user_id check
    user_input = st.chat_input("Type your message here...")

    # Process user input - CORRECTLY INDENTED
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_user_chat(user_id, st.session_state.messages)
        
        # Extract context from previous messages
        context = None
        for i in range(len(st.session_state.messages)-2, -1, -1):
            if st.session_state.messages[i]["role"] == "assistant" and "metadata" in st.session_state.messages[i]:
                context = st.session_state.messages[i]["metadata"]
                break
        
        try:
            with st.status("Processing your question...", expanded=False) as status:
                if st.session_state.context == "general":
                    # Get LLM service
                    llm = get_llm_service()
                    
                    # Use LLM for general questions with better error handling
                    if llm["available"]:
                        try:
                            status.update(label="Getting career advice...", state="running")
                            # Add debugging to sidebar
                            st.sidebar.write(f"Processing query: {user_input[:30]}...")
                            
                            course_keywords = ["course", "courses", "learn", "learning", "tutorial", "study", "class"]
                            is_course_query = any(keyword in user_input.lower() for keyword in course_keywords)

                            interview_keywords = ["interview", "prepare", "preparation", "mock", "practice"]
                            is_interview_query = any(keyword in user_input.lower() for keyword in interview_keywords)

                            if is_interview_query:
                                response = llm["generate_interview_resources"](user_input, context)
                                #st.sidebar.success("✓ Interview resources generated")
                            elif is_course_query:
                                # Use the specialized course recommendation function
                                response = llm["generate_course_recommendations"](user_input)
                                #st.sidebar.success("✓ Course recommendations generated")
                            else:
                                # Check if it's a career question
                                is_career_question = True
                                if llm.get("is_career_question"):
                                    is_career_question = llm["is_career_question"](user_input, context)
                                
                                # Now use the is_career_question variable within this else block
                                if is_career_question:
                                    # Call the generate_career_advice function
                                    response = llm["generate_career_advice"](user_input, context)
                                else:
                                    response = ("I'm focused on helping with your career-related questions. "
                                            "Could you please ask me something about your career, job search, "
                                            "resume, interview preparation, or professional development?")
                        except Exception as e:
                            st.sidebar.error(f"Error: {str(e)}")
                            response = f"I encountered an error processing your request: {str(e)}"
                    else:
                        # Fallback if LLM service is not available
                        status.update(label="Using fallback response...", state="running")
                        response = ("I'm sorry, but my advanced response service is currently unavailable. "
                                   "Here's a general response: I'm happy to help with your career journey! "
                                   "What specific questions do you have today?")
                        #st.sidebar.warning("Using fallback response system")
                else:
                    # Handle different contexts
                    if st.session_state.context == "jobs":
                        try:
                            if context and context.get("type") == "job_listings":
                                # Debug info
                                #st.sidebar.write("Job context detected, checking for specific job mention...")
                                
                                # The user is asking about job listings already shown
                                llm = get_llm_service()
                                
                                # Get jobs from context
                                job_listings = context.get("jobs", [])
                                selected_job = None
                                
                                # Print available jobs for debugging
                                st.sidebar.write(f"Available jobs ({len(job_listings)}):")
                                for idx, job in enumerate(job_listings):
                                    job_title = job.get("title", "")
                                    job_company = job.get("company", "")
                                    st.sidebar.write(f"{idx+1}. {job_title} at {job_company}")
                                
                                # More flexible job matching - check multiple variations
                                user_input_lower = user_input.lower()
                                for job in job_listings:
                                    job_title = job.get("title", "").lower()
                                    job_company = job.get("company", "").lower()
                                    
                                    # Check various matching patterns
                                    if (job_title in user_input_lower and job_company in user_input_lower) or \
                                    (f"{job_title} at {job_company}" in user_input_lower) or \
                                    (job_title in user_input_lower and any(company_part in user_input_lower for company_part in job_company.split())) or \
                                    (any(word in user_input_lower for word in job_title.split()) and job_company in user_input_lower):
                                        selected_job = job
                                        st.sidebar.success(f"✓ Selected job: {job.get('title')} at {job.get('company')}")
                                        break
                                
                                if selected_job:
                                    # Job identified - provide specific response
                                    if llm["available"]:
                                        # Create enhanced context with job details
                                        job_context = {
                                            "type": "specific_job",
                                            "job": selected_job,
                                            "request": user_input
                                        }
                                        
                                        # Generate specific job response
                                        try:
                                            job_title = selected_job.get('title', '')
                                            job_company = selected_job.get('company', '')
                                            job_location = selected_job.get('location', '')
                                            job_type = selected_job.get('type', '')
                                            job_skills = ', '.join(selected_job.get('skills', []))
                                            
                                            # Build detailed prompt with job info
                                            job_prompt = f"""
                                            The user is asking about this specific job: {job_title} at {job_company}.
                                            
                                            JOB DETAILS:
                                            - Title: {job_title}
                                            - Company: {job_company}
                                            - Location: {job_location}
                                            - Type: {job_type}
                                            - Required Skills: {job_skills}
                                            
                                            Original user query: {user_input}
                                            
                                            Please provide detailed information about this role, including:
                                            1. A summary of the position and its responsibilities
                                            2. Important requirements and qualifications
                                            3. Tips for applying successfully
                                            4. Potential interview questions for this role
                                            """
                                            
                                            response = llm["generate_career_advice"](job_prompt, job_context)
                                            st.sidebar.success("✓ Generated specific job response")
                                        except Exception as e:
                                            st.sidebar.error(f"Error generating job response: {str(e)}")
                                            # Fallback for specific job
                                            response = f"""
                                            ## {job_title} at {job_company}
                                            
                                            This position is located in {job_location} and is a {job_type} role.
                                            
                                            ### Required Skills:
                                            {job_skills}
                                            
                                            This looks like a great match for your experience! The Backend Developer role typically involves designing server architecture, implementing APIs, and ensuring scalability.
                                            
                                            Would you like tips for preparing your application for this position?
                                            """
                                    else:
                                        # Fallback for specific job if LLM not available
                                        job_title = selected_job.get('title', '')
                                        job_company = selected_job.get('company', '')
                                        job_location = selected_job.get('location', '')
                                        job_skills = ', '.join(selected_job.get('skills', []))
                                        
                                        response = f"""
                                        ## {job_title} at {job_company}
                                        
                                        This position is located in {job_location}.
                                        
                                        ### Required Skills:
                                        {job_skills}
                                        
                                        This looks like a great match for your experience! Would you like tips for preparing your application?
                                        """
                                else:
                                    # No specific job identified
                                    st.sidebar.warning("No specific job identified in user query")
                                    if "application" in user_input_lower or "apply" in user_input_lower or "resume" in user_input_lower:
                                        response = """
                                        # Application Tips
                                        
                                        Here are some general tips for job applications:
                                        
                                        1. **Customize your resume** for each application
                                        2. **Research the company** thoroughly before applying
                                        3. **Highlight relevant skills** and experience in your cover letter
                                        4. **Follow up** a week after submitting your application
                                        
                                        Would you like specific tips for a particular job?
                                        """
                                    else:
                                        # General job advice
                                        if llm["available"]:
                                            response = llm["generate_career_advice"](user_input, context)
                                        else:
                                            response = "Those are great jobs to consider! Would you like tips on preparing your application?"
                            else:
                                # Check if the user is looking for different types of jobs
                                job_search_keywords = ["looking for", "interested in", "searching for", "find", 
                                                    "seeking", "hunting", "need a job", "want a job", "job in"]
                                                    
                                if any(keyword in user_input.lower() for keyword in job_search_keywords):
                                    # Save user job preferences for future recommendations
                                    st.session_state.job_preferences = user_input
                                    
                                    # Use RAG to find relevant jobs based on the query
                                    rag = get_rag_service()
                                    if rag["available"]:
                                        # Extract job-related context if possible
                                        job_context = rag["search"](query=user_input)
                                        
                                    # Get personalized recommendations with the new context
                                    recommended_jobs = get_recommended_jobs(
                                        st.session_state.messages + [{"role": "user", "content": user_input}], 
                                        JOB_DATABASE
                                    )
                                    response = f"Based on your interest in {user_input}, here are some jobs that might be a good fit:\n\n"
                                    response += format_job_listings(recommended_jobs)
                                    
                                    # Add metadata for future context
                                    st.session_state.messages[-1]["metadata"] = {
                                        "type": "job_listings",
                                        "jobs": recommended_jobs,
                                        "user_preferences": user_input
                                    }
                                else:
                                    # Default to showing some recommended jobs
                                    recommended_jobs = get_recommended_jobs(st.session_state.messages, JOB_DATABASE)
                                    response = format_job_listings(recommended_jobs)
                        except Exception as e:
                            st.sidebar.error(f"Error in job processing: {str(e)}")
                            # Fallback to simple job recommendations
                            recommended_jobs = get_recommended_jobs(st.session_state.messages, JOB_DATABASE)
                            response = format_job_listings(recommended_jobs)
                    elif st.session_state.context == "resume":
                        # Check if we're in resume building mode
                        if st.session_state.resume_mode:
                            # We're already in resume building mode, process answers
                            current_q = st.session_state.current_question
                            current_field = resume_fields[current_q]
                            
                            # Store user's answer
                            st.session_state.user_info[current_field] = user_input
                            
                            # Move to next question or finish
                            st.session_state.current_question += 1
                            
                            if st.session_state.current_question < len(resume_questions):
                                # Ask next question
                                next_question = resume_questions[st.session_state.current_question]
                                response = f"Thanks! {next_question}"
                            else:
                                # All questions answered, generate resume
                                
                                try:
                                    with st.spinner("Generating your resume..."):
                                        try:
                                            # Print debug info
                                            print("User info for resume generation:")
                                            for key, val in st.session_state.user_info.items():
                                                print(f"{key}: {val}")
                                            
                                            # Generate resume content
                                            resume_data = generate_resume_content(st.session_state.user_info)
                                            
                                            # Verify resume data structure
                                            if not isinstance(resume_data, dict) or "header" not in resume_data:
                                                raise ValueError("Invalid resume data structure generated")
                                            
                                            print("Resume data generated successfully")
                                            st.session_state.resume_data = resume_data
                                            
                                            # Create PDF with better error handling
                                            pdf_path = create_pdf_resume(resume_data)
                                            if not os.path.exists(pdf_path):
                                                raise FileNotFoundError(f"PDF file was not created at {pdf_path}")
                                            
                                            print(f"PDF created at: {pdf_path}")
                                            st.session_state.pdf_path = pdf_path
                                            
                                            # Create download link
                                            pdf_filename = f"resume_{st.session_state.user_info['name'].replace(' ', '_')}.pdf"
                                            download_link = get_pdf_download_link(pdf_path, pdf_filename)
                                            
                                            # Add message with download link
                                            response = f"Great! I've created your resume based on the information you provided. Here's your resume PDF:<br><br>{download_link}<br><br>Is there anything else you'd like me to help you with?"
                                            
                                        except Exception as e:
                                            import traceback
                                            print(f"Resume generation error: {str(e)}")
                                            print(traceback.format_exc())  # Print full stack trace
                                            response = f"I encountered an error while generating your resume: {str(e)}. Please try again with more complete information."
                                            st.error(f"Resume generation error: {str(e)}")
                                        
                                        # Reset resume mode
                                        st.session_state.resume_mode = False
                                except Exception as outer_e:
                                    response = f"Unexpected error in resume processing: {str(outer_e)}"
                                    st.error(response)
                                    with st.spinner("Generating your resume..."):
                                        # Generate resume content
                                        resume_data = generate_resume_content(st.session_state.user_info)
                                        st.session_state.resume_data = resume_data
                                        
                                        # Create PDF
                                        pdf_path = create_pdf_resume(resume_data)
                                        st.session_state.pdf_path = pdf_path
                                        
                                        # Create download link
                                        pdf_filename = f"resume_{st.session_state.user_info['name'].replace(' ', '_')}.pdf"
                                        download_link = get_pdf_download_link(pdf_path, pdf_filename)
                                        
                                        # Add message with download link
                                        response = f"Great! I've created your resume based on the information you provided. Here's your resume PDF:<br><br>{download_link}<br><br>Is there anything else you'd like me to help you with?"
                                        
                                        # Reset resume mode
                                        st.session_state.resume_mode = False
                                except Exception as e:
                                    response = f"I encountered an error while generating your resume: {str(e)}. Please try again or check your API key configuration."
                                    st.error(f"Resume generation error: {str(e)}")
                        else:
                            # Check if user wants to build a resume
                            if any(keyword in user_input.lower() for keyword in ["resume", "build", "cv", "create"]):
                                # Start resume building mode
                                st.session_state.resume_mode = True
                                st.session_state.current_question = 0
                                
                                # Send first question
                                response = f"Great! I'll help you build your resume. Let's start with some basic information. {resume_questions[0]}"
                            else:
                                # Regular resume advice
                                llm = get_llm_service()
                                if llm["available"]:
                                    context = {"type": "resume_advice"}
                                    response = llm["generate_career_advice"](user_input, context)
                                else:
                                    response = "For resume help, I recommend:\n1. Using clear formatting\n2. Highlighting achievements with numbers\n3. Keeping it concise and relevant\n4. Customizing for each application\n\nWould you like to build a resume now? Just say 'create resume' to begin."

                    elif st.session_state.context == "interview":
                        # Get LLM service
                        llm = get_llm_service()
                        
                        # Use LLM for interview questions with context-specific handling
                        if llm["available"]:
                            try:
                                status.update(label="Preparing interview advice...", state="running")
                                # Add debugging to sidebar
                                st.sidebar.write(f"Processing interview query: {user_input[:30]}...")
                                
                                # Create interview-specific context
                                interview_context = {
                                    "type": "interview_advice",
                                    "area": "interview preparation",
                                    "focus": "providing structured responses and practical tips"
                                }
                                
                                # Enhance the user input with interview context if needed
                                enhanced_query = f"Interview question: {user_input}"
                                
                                # Call the generate_career_advice function with interview context
                                response = llm["generate_career_advice"](enhanced_query, interview_context)
                                st.sidebar.success("✓ Interview response generated successfully")
                            except Exception as e:
                                st.sidebar.error(f"Interview advice error: {str(e)}")
                                response = f"I encountered an error processing your interview question: {str(e)}"
                        else:
                            # Fallback if LLM service is not available
                            status.update(label="Using fallback interview response...", state="running")
                            response = ("I'm happy to help with interview preparation! For most interviews, I recommend: "
                                    "1. Research the company thoroughly before the interview\n"
                                    "2. Prepare examples that demonstrate your skills and experience\n"
                                    "3. Practice the STAR method (Situation, Task, Action, Result) for behavioral questions\n"
                                    "4. Prepare thoughtful questions to ask the interviewer\n\n"
                                    "Would you like specific advice for technical or behavioral interviews?")
                            st.sidebar.warning("Using fallback interview response system")

                    elif st.session_state.context == "mentorship":
                        # Get LLM service
                        llm = get_llm_service()
                        
                        # Use LLM for mentorship advice with context-specific handling
                        if llm["available"]:
                            try:
                                status.update(label="Finding mentorship advice...", state="running")
                                # Add debugging to sidebar
                                st.sidebar.write(f"Processing mentorship query: {user_input[:30]}...")
                                
                                # Create mentorship-specific context
                                mentorship_context = {
                                    "type": "mentorship_advice",
                                    "area": "professional mentorship",
                                    "focus": "providing guidance on finding and working with mentors"
                                }
                                
                                # Enhance user input with mentorship focus
                                enhanced_query = f"Mentorship question: {user_input}"
                                
                                # Call the generate_career_advice function with mentorship context
                                response = llm["generate_career_advice"](enhanced_query, mentorship_context)
                                st.sidebar.success("✓ Mentorship response generated successfully")
                            except Exception as e:
                                st.sidebar.error(f"Mentorship advice error: {str(e)}")
                                response = f"I encountered an error processing your mentorship question: {str(e)}"
                        else:
                            # Fallback if LLM service is not available
                            status.update(label="Using fallback mentorship response...", state="running")
                            response = ("Finding the right mentor can be transformative for your career! Here are some tips:\n"
                                    "1. Identify what specific guidance you're seeking\n"
                                    "2. Look within your current organization and extended network\n"
                                    "3. Attend industry events and join professional communities\n"
                                    "4. Be specific about your goals when approaching potential mentors\n\n"
                                    "What industry or specific skills are you looking to develop with a mentor?")
                            st.sidebar.warning("Using fallback mentorship response system")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            response = "I apologize, but I encountered an unexpected issue. Please try again with a different question."
        
        response_msg = {
            "role": "assistant",
            "content": response
        }

        # Preserve metadata when this is a job-related response
        if st.session_state.context == "jobs" and context and context.get("type") == "job_listings":
            response_msg["metadata"] = {
                "type": "job_listings",
                "jobs": context.get("jobs", [])
            }
            
        st.session_state.messages.append(response_msg)
        
        # Save updated chat history
        save_user_chat(user_id, st.session_state.messages)
        
        st.rerun()

    # Display welcome message for new users
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": CONTEXT_OPTIONS[st.session_state.context]["welcome"]
        })
        save_user_chat(user_id, st.session_state.messages)
        st.rerun()
else:
    # If no user_id and not showing login, show a message
    st.info("Please log in to use the chatbot.")