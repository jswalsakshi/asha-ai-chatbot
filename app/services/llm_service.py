import os
import json
import google.generativeai as genai
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Import LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load environment variables
load_dotenv()

# API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Direct Gemini API setup as a fallback option
genai.configure(api_key=GEMINI_API_KEY)

# Career-focused system prompt
SYSTEM_PROMPT = """
You are Asha, a helpful career assistant. Your goal is to provide clear, practical advice on career development, 
job searching, resume building, interviewing, and professional growth. Remember:
- Be supportive, positive, and empowering
- Provide specific, actionable advice
- Focus ONLY on helping people advance their careers, job searching, and professional development
- Be inclusive and consider diverse backgrounds and career paths
- Cite statistics or best practices when relevant
- If asked about non-career topics, politely redirect the conversation back to career advice
- NEVER provide advice on topics unrelated to careers, jobs, professional development, or workplace dynamics
- When discussing specific job listings, provide detailed information and personalized application advice
"""

# Initialize conversation memory
memory = ConversationBufferMemory(return_messages=True)

# Track if we've verified the API key
api_key_verified = False

def verify_api_key():
    """Verify that the API key is valid and working"""
    global api_key_verified
    
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        return False
    
    # Only verify once
    if api_key_verified:
        return True
    
    try:
        # Test with a simple prompt
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Say hello")
        if response and hasattr(response, 'text'):
            api_key_verified = True
            print("API key verified successfully")
            return True
        return False
    except Exception as e:
        print(f"API key verification failed: {str(e)}")
        return False

def get_conversation_chain():
    """
    Initialize and return the conversation chain with memory
    
    Returns:
        ConversationChain or None: The initialized conversation chain, or None if failed
    """
    # Verify API key first
    if not verify_api_key():
        return None
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.7,
            max_output_tokens=800,
        )
        
        # Create a prompt template with system message and conversation history
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="{input}")
        ])
        
        # Create the conversation chain
        chain = ConversationChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=False  # Set to True for debugging
        )
        
        return chain
    except Exception as e:
        print(f"Error creating conversation chain: {str(e)}")
        return None

def direct_generate_response(query: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate response directly using the Google Generative AI library as a fallback
    
    Args:
        query: User's question
        context: Optional context information about previous interactions
    
    Returns:
        str: Generated response
    """
    try:
        if not verify_api_key():
            return "I'm experiencing connectivity issues. Please check your API key configuration."
        
        # Create conversation context with system prompt and past messages
        conversation = [SYSTEM_PROMPT]
        
        # Add memory context if available
        for msg in memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                conversation.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                conversation.append(f"Assistant: {msg.content}")
        
        # Add context information if available
        if context:
            if context.get("type") == "job_listings" and context.get("jobs"):
                jobs = context.get("jobs")
                job_info = "\n".join([f"Job: {job.get('title')} at {job.get('company')}" for job in jobs[:3] if job.get('title')])
                conversation.append(f"Context: We were discussing these job listings: {job_info}")
        
        # Add the current question
        conversation.append(f"User: {query}")
        conversation.append("Assistant: ")
        
        # Generate content with full context
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("\n".join(conversation))
        
        if response and hasattr(response, 'text'):
            # Add to memory for future context
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(response.text)
            
            return response.text
        
        return "I couldn't generate a response at this time."
        
    except Exception as e:
        print(f"Error in direct_generate_response: {str(e)}")
        return f"I encountered an issue with the response generation. Please try again with a different question."

def generate_career_advice(query: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate career advice using LangChain with fallback options
    
    Args:
        query (str): User's question
        context (dict, optional): Additional context about previous interactions
    
    Returns:
        str: Generated response or redirection message if off-topic
    """
    try:
        print(f"Processing query: {query}")
        print(f"Context: {context}")
        
        # Special handling for job listings context - these are always job-related
        if context and context.get("type") == "job_listings":
            print("Job listing context detected")
            # Check if this looks like a specific job interest
            job_interest_keywords = [
                "interested in", "tell me about", "like", "want", "apply", "yes", "more about",
                "details", "salary", "requirements", "responsibilities", "company", "location",
                "remote", "hybrid", "skills", "qualifications", "application", "sounds good"
            ]
            
            if any(keyword in query.lower() for keyword in job_interest_keywords):
                print("User expressing job interest, generating job details response")
                return generate_job_details_response(query, context.get("jobs", []))
        
        # For other queries, check if it's a career question
        if not is_career_question(query, context):
            print("Not a career question")
            return ("I'm designed to assist with career-related questions, job searching, resume building, "
                    "interviews, mentorship, and professional development. Could you please ask me something "
                    "related to these areas so I can help you with your professional journey?")
        
        print("Career question detected, continuing...")
        
        # Process context if available
        if context:
            print(f"Injecting context: {context.get('type', 'unknown')}")
            inject_context_into_memory(context)
        
        # Handle follow-up questions specially - this is now redundant with the check above
        # but keeping for safety in case the first check is missed
        if is_followup_question(query) and context and context.get("type") == "job_listings":
            print("Handling job listing follow-up")
            return generate_job_details_response(query, context.get("jobs", []))
        
        print("Getting conversation chain...")
        # Try LangChain implementation first
        chain = get_conversation_chain()
        
        if chain:
            try:
                # Add user query to memory
                memory.chat_memory.add_user_message(query)
                
                # Generate response using the chain
                print("Generating response with LangChain...")
                response = chain.predict(input=query)
                
                # Record AI response in memory
                memory.chat_memory.add_ai_message(response)
                
                return response.strip()
            except Exception as chain_error:
                print(f"Chain error: {str(chain_error)}")
                # Fall back to direct API if chain fails
                print("Falling back to direct API...")
                return direct_generate_response(query, context)
        else:
            # Chain failed to initialize, use direct API
            print("Using direct API (chain unavailable)...")
            return direct_generate_response(query, context)
            
    except Exception as e:
        print(f"Error in generate_career_advice: {str(e)}")
        # Try direct API as last resort
        try:
            return direct_generate_response(query, context)
        except:
            return "I'm experiencing technical difficulties. Let me offer some general career advice: networking, continuous learning, and building a strong personal brand are key to career success."

def inject_context_into_memory(context: Dict[str, Any]) -> None:
    """
    Inject contextual information into the conversation memory
    
    Args:
        context (dict): Context information to inject
    """
    try:
        if context.get("type") == "job_listings" and context.get("jobs"):
            job_titles = [f"{job.get('title', 'a job')} at {job.get('company', 'a company')}" for job in context.get("jobs", [])]
            job_context_msg = f"We were discussing these jobs: {', '.join(job_titles[:3])}"
            
            # Add as AI message to memory
            memory.chat_memory.add_ai_message(job_context_msg)
            print(f"Added context to memory: {job_context_msg}")
            
            # Also add user preferences if available
            if "user_preferences" in context:
                pref_msg = f"You mentioned interest in: {context.get('user_preferences')}"
                memory.chat_memory.add_ai_message(pref_msg)
                print(f"Added preferences to memory: {pref_msg}")
    except Exception as e:
        print(f"Error injecting context: {str(e)}")

def is_followup_question(query: str) -> bool:
    """
    Determine if a query is likely a follow-up question
    
    Args:
        query (str): User's question
    
    Returns:
        bool: True if it's a follow-up question
    """
    query_lower = query.lower()
    followup_phrases = [
        "tell me more", "more information", "more details", "explain further",
        "elaborate", "about this", "about that", "more about", "what else",
        "go on", "continue", "and then", "what about", "how about", "can you explain",
        "yes", "interested", "sounds good", "great", "ok", "okay", "sure",
        "why", "how", "when", "where", "who", "which one", "please", "thanks"
    ]
    
    # Check for short questions that might be follow-ups
    is_short = len(query.split()) <= 8  # Increased from 6 to catch more follow-ups
    has_followup_phrase = any(phrase in query_lower for phrase in followup_phrases)
    
    # More permissive criteria - either short or contains follow-up phrase
    return is_short or has_followup_phrase

# Modify the generate_job_details_response function around line 222

def generate_job_details_response(query: str, jobs: List[Dict[str, Any]]) -> str:
    """
    Generate a detailed response about jobs based on the query
    
    Args:
        query (str): User's follow-up question
        jobs (list): List of job details
    
    Returns:
        str: Formatted response with job details
    """
    print(f"Generating job details for: {query}")
    print(f"Jobs available: {len(jobs)}")
    print(f"Current jobs in context: {[job.get('title', 'Unknown') for job in jobs]}")
    
    if not jobs:
        return "I don't have any job details to share at the moment. Would you like me to help you find job opportunities?"
    
    # Try to identify which specific job the user is asking about
    query_lower = query.lower()
    selected_job = None
    selected_idx = 0
    
    # First check: Look for job title or company mentions in the query
    for idx, job in enumerate(jobs):
        title = job.get('title', '').lower()
        company = job.get('company', '').lower()
        
        # Check if job title or company is mentioned in the query
        if (title and title in query_lower) or (company and company in query_lower):
            selected_job = job
            selected_idx = idx
            print(f"Found specific job match by name: {title} at {company}")
            break
    
    # Second check: If user responded with "yes", "sure", etc., use the first job
    # This is the case that was failing before
    if not selected_job and (
        query_lower in ["yes", "sure", "okay", "ok", "please", "definitely", "absolutely"] or
        any(affirmation in query_lower for affirmation in ["yes", "want", "like to", "interested", "tell me more"])
    ):
        # For simple affirmative responses, keep the job that was asked about most recently
        # We'll assume it's the first one in the list as that's what we showed
        selected_job = jobs[0]
        selected_idx = 0
        print(f"User affirmed interest - using first job: {selected_job.get('title', 'Unknown job')}")
    
    # Fallback: If still no job selected, use the first one
    if not selected_job:
        selected_job = jobs[0]
        print(f"No specific match, using first job by default: {selected_job.get('title', 'Unknown job')}")
    
    # The rest of the existing function...
    job_title = selected_job.get('title', 'Job Position')
    
    # For simple "yes" responses to application tips question, provide application tips
    if query_lower in ["yes", "sure", "okay", "ok", "please", "definitely", "absolutely", "would like", "i want"]:
        response = f"### Application Tips for {job_title} at {selected_job.get('company', 'the company')}\n\n"
        response += "1. **Research the company:** Understand Microsoft's products, culture, and recent news to show your genuine interest.\n\n"
        response += "2. **Tailor your resume:** Highlight your experience with "
        
        # Add skill-specific advice if skills are available
        if selected_job.get('skills'):
            skills = selected_job.get('skills')
            if isinstance(skills, list) and skills:
                response += f"{', '.join(skills[:3])}. "
            elif isinstance(skills, str):
                response += f"{skills}. "
        else:
            response += "relevant technologies and frameworks. "
        
        response += "\n\n3. **Prepare technical examples:** Be ready to discuss specific projects where you've used similar technologies.\n\n"
        response += "4. **Practice coding questions:** For technical roles, review data structures, algorithms, and be prepared for coding exercises.\n\n"
        response += "5. **Prepare questions:** Have thoughtful questions about the team, projects, and growth opportunities.\n\n"
        
        response += f"Would you like more specific advice for preparing for interviews at {selected_job.get('company', 'the company')}?"
        return response
        
    # Build a comprehensive response for non-application questions
    response = f"### {job_title}\n\n"
    
    # Rest of your existing response building code...
    # (Keep all the existing code for building the job details response)
    
    return response

def is_career_question(query: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Determine if a query is career-related using a comprehensive topic classification approach
    
    Args:
        query (str): User query
        context (dict, optional): Additional context from previous messages
    
    Returns:
        bool: True if career-related, False otherwise
    """
    # IMPORTANT: Always consider job follow-ups as career questions
    if context and context.get("type") == "job_listings":
        # If we have job context, most responses are likely job-related
        return True
    
    # Check for job interest phrases that might not be caught by keywords
    job_interest_phrases = [
        "interested in", "tell me more about", "more about the", "like to know more", 
        "want to apply", "how to apply", "yes", "please", "sure", "sounds good",
        "like the", "would like", "about this job", "about that job", "about the position",
        "want to learn", "job description", "requirements", "qualifications", "salary",
        "sounds interesting", "company", "location", "remote", "hybrid", "onsite",
        "prepare for", "apply for", "perfect", "good fit", "skills needed"
    ]
    
    query_lower = query.lower()
    if any(phrase in query_lower for phrase in job_interest_phrases):
        return True
    
    # Always allow follow-up questions in an ongoing conversation
    if is_followup_question(query) and memory.chat_memory.messages:
        return True
    
    # Define relevant topics and their keywords (more comprehensive)
    relevant_topics = {
        "career_development": [
            "career", "profession", "professional", "growth", "advance", "progress", "path", 
            "trajectory", "promotion", "transition", "pivot", "change", "switch", "move", 
            "advancement", "aspiration", "goal", "objective", "ambition", "direction", "success",
            "industry", "field", "sector", "domain", "area", "discipline", "specialty"
        ],
        "job_search": [
            "job", "work", "employ", "position", "role", "opening", "vacancy", "hiring", 
            "application", "apply", "workplace", "company", "organization", "corporation", 
            "enterprise", "firm", "business", "startup", "remote", "hybrid", "virtual", "office",
            "job market", "job board", "job site", "job portal", "job platform", "job opportunity",
            "job listing", "job posting", "job offer", "opportunity", "prospect", "opening"
        ],
        "compensation": [
            "salary", "compensation", "pay", "wage", "income", "earning", "remuneration", "stipend", 
            "benefit", "bonus", "raise", "increment", "hike", "stock", "equity", "option", "esop",
            "negotiate", "bargain", "discussion", "package", "offer", "counter-offer", "money"
        ],
        "resume_portfolio": [
            "resume", "cv", "curriculum vitae", "bio", "biography", "portfolio", "profile", 
            "experience", "background", "history", "track record", "qualification", "credential",
            "achievement", "accomplishment", "education", "degree", "diploma", "certification",
            "certificate", "training", "skill", "competency", "ability", "capability", "expertise",
            "proficiency", "mastery", "strength", "talent", "aptitude", "flair", "document"
        ],
        "interview_process": [
            "interview", "recruiter", "hiring manager", "talent acquisition", "hr", "human resources",
            "question", "answer", "preparation", "behavioral", "technical", "screening", "assessment", 
            "evaluation", "round", "stage", "phase", "onsite", "offsite", "phone screen", "video call",
            "panel", "one-on-one", "group", "case study", "presentation", "test", "quiz", "challenge"
        ],
        "networking": [
            "mentor", "guide", "coach", "advice", "guidance", "support", "recommendation", 
            "suggestion", "feedback", "networking", "connection", "contact", "referral", 
            "introduction", "linkedin", "social media", "professional network", "community",
            "relationship", "alliance", "association", "group", "meetup", "conference", "event",
            "workshop", "webinar", "seminar", "symposium", "forum", "panel", "discussion"
        ],
        "workplace_dynamics": [
            "colleague", "coworker", "peer", "supervisor", "manager", "boss", "subordinate", "team",
            "collaboration", "cooperation", "coordination", "conflict", "resolution", "problem", 
            "solution", "leadership", "management", "culture", "environment", "atmosphere", "climate",
            "politics", "communication", "feedback", "review", "performance", "productivity",
            "efficiency", "effectiveness", "work-life", "balance", "burnout", "stress", "pressure",
            "wellbeing", "wellness", "health", "satisfaction", "fulfillment", "enjoyment", "happiness"
        ],
        "skill_development": [
            "learn", "study", "training", "development", "improvement", "enhancement", "up-skilling",
            "reskilling", "course", "program", "degree", "education", "bootcamp", "certification",
            "workshop", "class", "lecture", "seminar", "conference", "knowledge", "expertise",
            "competency", "capability", "ability", "talent", "strength", "aptitude", "intelligence",
            "hard skill", "soft skill", "technical skill", "people skill", "communication skill"
        ],
        "job_specific": [
            "developer", "engineer", "designer", "manager", "director", "analyst", "specialist",
            "consultant", "coordinator", "assistant", "associate", "lead", "senior", "junior",
            "intern", "ceo", "cto", "cfo", "vp", "head", "chief", "president", "founder"
        ]
    }
    
    # Check if the query contains any relevant keywords
    for topic, keywords in relevant_topics.items():
        if any(keyword.lower() in query_lower for keyword in keywords):
            return True
    
    # Additional check for general career questions that might not contain specific keywords
    general_career_phrases = [
        "how to get a", "how to become", "what should i do", "i need help with my", 
        "looking for advice", "need guidance", "struggling with", "tips for", 
        "best practices", "ways to improve", "how do i prepare", "what's the best way",
        "i'm interested in", "i want to work", "i'm applying", "i want to be", 
        "i want to become", "help me with", "how can i", "should i", "would it be",
        "is it good", "is it bad", "recommend", "suggest", "advice", "thoughts on"
    ]
    
    if any(phrase in query_lower for phrase in general_career_phrases):
        return True
    
    # Even more lenient - allow anything related to companies
    company_terms = ["google", "microsoft", "apple", "amazon", "meta", "facebook", "twitter",
                    "linkedin", "startup", "corporation", "firm", "enterprise", "organization",
                    "employer", "tech", "company", "business", "industry"]
    
    if any(term in query_lower for term in company_terms):
        return True
    
    # Lastly, any references to programming, coding or tech as they're likely career-related in this context
    tech_terms = ["coding", "programming", "software", "web", "app", "development", "engineering",
                 "javascript", "python", "java", "c++", "html", "css", "react", "node", "sql",
                 "database", "cloud", "ai", "ml", "data", "frontend", "backend", "fullstack"]
    
    if any(term in query_lower for term in tech_terms):
        return True
    
    return False

def reset_conversation():
    """Clear the conversation memory"""
    global memory
    memory = ConversationBufferMemory(return_messages=True)