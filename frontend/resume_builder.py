import streamlit as st
import google.generativeai as genai
from fpdf import FPDF
import base64
import json
import os
from datetime import datetime

# Configure your Gemini API key
def configure_genai(api_key):
    genai.configure(api_key=api_key)

# Generate resume content using Gemini
def generate_resume_content(user_info):
    for key, value in user_info.items():
        print(f"Field '{key}': {value[:50]}..." if value else f"Field '{key}' is empty")
        
        # Ensure no field is completely empty
        if not value and key in ['name', 'email']:
            user_info[key] = f"[Please add your {key}]"
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Debug: Log what's being sent to the model
    print(f"Work experience being sent to model: {user_info.get('experience', '')}")
    
    prompt = f"""
    Create a professional resume based on the following information:
    
    Personal Information:
    - Name: {user_info.get('name', '')}
    - Email: {user_info.get('email', '')}
    - Phone: {user_info.get('phone', '')}
    - Location: {user_info.get('location', '')}
    
    Professional Summary:
    {user_info.get('summary', '')}
    
    Education:
    {user_info.get('education', '')}
    
    Work Experience:
    {user_info.get('experience', '')}
    
    Skills:
    {user_info.get('skills', '')}
    
    Format the resume into clear sections. Keep the content professional and concise.
    IMPORTANT: Preserve the full job titles and company names exactly as provided.
    Return the result as a JSON object with the following keys:
    "header" (containing name, contact info),
    "summary",
    "education" (list of education entries),
    "experience" (list of work experiences with full position and company names),
    "skills" (list of skills)
    """
    
    response = model.generate_content(prompt)
    try:
        # Try to parse JSON directly from the response
        content = response.text
        # Find JSON content between triple backticks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        resume_data = json.loads(content)
        return resume_data
    except json.JSONDecodeError:
        # If parsing fails, try to extract structured data manually
        st.error("Error parsing AI response. Falling back to manual formatting.")
        return {
            "header": {
                "name": user_info.get('name', ''),
                "email": user_info.get('email', ''),
                "phone": user_info.get('phone', ''),
                "location": user_info.get('location', '')
            },
            "summary": user_info.get('summary', ''),
            "education": [{"entry": user_info.get('education', '')}],
            "experience": [{"entry": user_info.get('experience', '')}],
            "skills": user_info.get('skills', '').split(", ")
        }

# Create PDF resume
def create_pdf_resume(resume_data):
    pdf = FPDF()
    pdf.add_page()
    
    # Set colors
    header_color = (137, 80, 107)  # #89506B in RGB
    text_color = (255, 255, 255)   # White text
    
    # Create colored header strip
    pdf.set_fill_color(*header_color)
    pdf.rect(0, 0, 210, 30, 'F')  # Draw a filled rectangle at the top
    
    # Header text in white
    pdf.set_text_color(*text_color)
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 15, resume_data["header"].get("name", ""), 0, 1, "C")
    
    # Contact information in white, centered under name
    pdf.set_font("Arial", "", 10)
    
    # Create a single line of contact information
    contact_info = []
    if resume_data['header'].get('email', ''):
        contact_info.append(resume_data['header'].get('email', ''))
    if resume_data['header'].get('phone', ''):
        contact_info.append(resume_data['header'].get('phone', ''))
    if resume_data['header'].get('location', ''):
        contact_info.append(resume_data['header'].get('location', ''))
    
    pdf.cell(0, 5, " | ".join(contact_info), 0, 1, "C")
    
    # Reset text color to black for the rest of the document
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    # Summary
    pdf.set_font("Arial", "B", 12)
    pdf.set_fill_color(*header_color)
    pdf.set_text_color(*text_color)
    pdf.cell(0, 8, "Professional Summary", 0, 1, "L", 1)
    pdf.set_text_color(0, 0, 0)  # Reset to black text
    pdf.ln(2)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, resume_data.get("summary", ""))
    pdf.ln(5)
    
    # Education
    pdf.set_font("Arial", "B", 12)
    pdf.set_fill_color(*header_color)
    pdf.set_text_color(*text_color)
    pdf.cell(0, 8, "Education", 0, 1, "L", 1)
    pdf.set_text_color(0, 0, 0)  # Reset to black text
    pdf.ln(2)
    pdf.set_font("Arial", "", 10)
    
    for edu in resume_data.get("education", []):
        if isinstance(edu, dict):
            entry = edu.get("entry", "")
            if not entry:
                # Ensure we include all education information
                degree = edu.get("degree", "")
                institution = edu.get("institution", "")
                year = edu.get("year", "")
                entry = f"{degree}"
                if institution:
                    entry += f" - {institution}"
                if year:
                    entry += f", {year}"
        else:
            entry = str(edu)
        pdf.multi_cell(0, 5, entry)
        pdf.ln(2)
    pdf.ln(3)
    
    # Experience
    pdf.set_font("Arial", "B", 12)
    pdf.set_fill_color(*header_color)
    pdf.set_text_color(*text_color)
    pdf.cell(0, 8, "Work Experience", 0, 1, "L", 1)
    pdf.set_text_color(0, 0, 0)  # Reset to black text
    pdf.ln(2)
    pdf.set_font("Arial", "", 10)
    
    for exp in resume_data.get("experience", []):
        if isinstance(exp, dict):
            if "entry" in exp:
                # Use the complete entry text without truncation
                full_entry = exp["entry"]
                pdf.multi_cell(0, 5, full_entry)
            else:
                # Ensure we're capturing the full position and company text
                position = exp.get("position", "")
                company = exp.get("company", "")
                period = exp.get("period", "")
                description = exp.get("description", "")
                
                # Format the job title line, preserving the complete position text
                job_line = f"{position}"
                if company:
                    job_line += f" at {company}"
                if period:
                    job_line += f" ({period})"
                    
                pdf.set_font("Arial", "B", 10)
                pdf.cell(0, 5, job_line, 0, 1)
                pdf.set_font("Arial", "", 10)
                pdf.multi_cell(0, 5, description)
        else:
            # If it's a simple string, print the full string
            pdf.multi_cell(0, 5, str(exp))
        pdf.ln(2)
    pdf.ln(3)
    
    # Skills
    pdf.set_font("Arial", "B", 12)
    pdf.set_fill_color(*header_color)
    pdf.set_text_color(*text_color)
    pdf.cell(0, 8, "Skills", 0, 1, "L", 1)
    pdf.set_text_color(0, 0, 0)  # Reset to black text
    pdf.ln(2)
    pdf.set_font("Arial", "", 10)
    
    skills = resume_data.get("skills", [])
    if isinstance(skills, list):
        skill_text = ", ".join(skills)
    else:
        skill_text = str(skills)
    
    pdf.multi_cell(0, 5, skill_text)
    
    # Add a footer with Asha branding
    pdf.set_y(-15)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(*header_color)
    pdf.cell(0, 10, f"Generated by Asha AI Career Assistant - {datetime.now().strftime('%Y-%m-%d')}", 0, 0, "C")
    
    # Save to a buffer
    import tempfile
    temp_dir = tempfile.gettempdir()
    pdf_output = os.path.join(temp_dir, f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    pdf.output(pdf_output)
    
    return pdf_output

# Function to create a download link
def get_pdf_download_link(pdf_path, filename):
    try:
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        
        # Make sure we're getting valid data
        if not pdf_data or len(pdf_data) < 100:
            raise ValueError(f"PDF data too small ({len(pdf_data)} bytes)")
            
        b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
        download_link = f'''
        <a href="data:application/pdf;base64,{b64_pdf}" 
           download="{filename}" 
           style="display:inline-block; padding:10px 20px; background-color:#89506B; 
                  color:white; text-decoration:none; border-radius:5px; 
                  font-weight:bold; margin-top:10px; text-align:center;">
            📄 Download Resume PDF
        </a>
        '''
        return download_link
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<p style='color:red'>Error creating download link: {str(e)}</p>"

# Main chatbot application
def main():
    st.set_page_config(page_title="Resume Builder Chatbot", page_icon="📄")
    
    st.title("Resume Builder Chatbot")
    
    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter your Gemini API Key", type="password")
        if api_key:
            configure_genai(api_key)
            st.success("API Key configured!")
        else:
            st.warning("Please enter your Gemini API key")
    
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your resume building assistant. I can help you create a professional resume. Would you like to build a resume today?"}
        ]
    
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

    # Updated resume questions sequence with clearer instructions
    resume_questions = [
        "What is your full name?",
        "What is your email address?",
        "What is your phone number?",
        "What is your location (City, State/Country)?",
        "Please write a brief professional summary (2-3 sentences about your background and career goals).",
        "Please share your education details (Degree, Institution, Year).",
        "Please describe your work experience including complete job titles and companies (e.g. 'Data Scientist at Google, 2020-2023').",
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
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "<a href" in message["content"]:
                st.markdown(message["content"], unsafe_allow_html=True)
            else:
                st.write(message["content"])
    
    # User input
    user_input = st.chat_input("Type a message...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Process user input
        if not st.session_state.resume_mode:
            # Check if user wants to build a resume
            if any(keyword in user_input.lower() for keyword in ["resume", "build", "cv", "create"]):
                # Start resume building mode
                st.session_state.resume_mode = True
                st.session_state.current_question = 0
                
                # Send first question
                assistant_response = f"Great! I'll help you build your resume. Let's start with some basic information. {resume_questions[0]}"
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
                # Display assistant message
                with st.chat_message("assistant"):
                    st.write(assistant_response)
            else:
                # Regular chat mode with Gemini
                if api_key:
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    # Prepare conversation history for context
                    conversation_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1]])
                    prompt = f"Conversation history:\n{conversation_history}\n\nUser: {user_input}\n\nRespond as a helpful assistant. Keep responses concise and friendly."
                    
                    response = model.generate_content(prompt)
                    assistant_response = response.text
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    
                    # Display assistant message
                    with st.chat_message("assistant"):
                        st.write(assistant_response)
                else:
                    # Respond without API key
                    assistant_response = "I need a Gemini API key to help you properly. Please add your API key in the sidebar."
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    with st.chat_message("assistant"):
                        st.write(assistant_response)
        else:
            # We're in resume building mode, process answer and ask next question
            current_q = st.session_state.current_question
            current_field = resume_fields[current_q]
            
            # Store user's answer (preserve complete text)
            st.session_state.user_info[current_field] = user_input.strip()
            
            # Move to next question or finish
            st.session_state.current_question += 1
            
            if st.session_state.current_question < len(resume_questions):
                # Ask next question
                next_question = resume_questions[st.session_state.current_question]
                assistant_response = f"Thanks! {next_question}"
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
                # Display assistant message
                with st.chat_message("assistant"):
                    st.write(assistant_response)
            else:
                # All questions answered, generate resume
                if api_key:
                    with st.spinner("Generating your resume..."):
                        try:
                            # Debug print before generation
                            print(f"Generating resume with experience: {st.session_state.user_info['experience']}")
                            
                            # Generate resume content using Gemini
                            resume_data = generate_resume_content(st.session_state.user_info)
                            st.session_state.resume_data = resume_data
                            
                            # Create PDF
                            pdf_path = create_pdf_resume(resume_data)
                            st.session_state.pdf_path = pdf_path
                            
                            # Create download link
                            pdf_filename = f"resume_{st.session_state.user_info['name'].replace(' ', '_')}.pdf"
                            download_link = get_pdf_download_link(pdf_path, pdf_filename)
                            
                            # Add message to chat history
                            assistant_response = f"Great! I've created your resume based on the information you provided. Here's your resume PDF:\n\n{download_link}\n\nIs there anything else you'd like me to help you with?"
                            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                            
                            # Display assistant message with HTML support for download link
                            with st.chat_message("assistant"):
                                st.markdown(assistant_response, unsafe_allow_html=True)
                        except Exception as e:
                            error_message = f"Error generating resume: {str(e)}"
                            st.error(error_message)
                            assistant_response = f"I encountered an error while creating your resume: {str(e)}. Please try again."
                            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                            with st.chat_message("assistant"):
                                st.write(assistant_response)
                        finally:
                            # Reset resume mode
                            st.session_state.resume_mode = False
                else:
                    # Respond without API key
                    assistant_response = "I need a Gemini API key to generate your resume. Please add your API key in the sidebar."
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    with st.chat_message("assistant"):
                        st.write(assistant_response)

# Run the app
if __name__ == "__main__":
    main()