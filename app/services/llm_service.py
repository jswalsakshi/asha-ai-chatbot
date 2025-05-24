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

# Enhanced career-focused system prompt with structured response guidance
SYSTEM_PROMPT = """
You are Asha, a helpful career assistant. Your goal is to provide clear, practical advice on career development, 
job searching, resume building, interviewing, and professional growth.

RESPONSE FORMATTING REQUIREMENTS:
- Format ALL responses in a highly structured, organized manner
- Use Markdown formatting consistently (headings, lists, tables)
- Create tables with clear headers for presenting comparative information
- Use bullet points or numbered lists for steps, advantages, or key points
- Break information into clear sections with ### or ## headings
- Keep responses concise, focused, and easy to scan
- Bold important terms or concepts using **bold** formatting
- Use summaries or TL;DR sections for longer responses

CONTENT GUIDELINES:
- Be supportive, positive, and empowering
- Provide specific, actionable advice
- Focus ONLY on helping people advance their careers and professional development
- Be inclusive and consider diverse backgrounds and career paths
- Cite statistics or best practices when relevant
- If asked about non-career topics, politely redirect the conversation
- NEVER provide advice on topics unrelated to careers, jobs, or professional development
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
        
        # Create a prompt template with enhanced system message for structured responses
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
    Generate structured response directly using the Google Generative AI library as a fallback
    
    Args:
        query: User's question
        context: Optional context information about previous interactions
    
    Returns:
        str: Generated structured response
    """
    try:
        if not verify_api_key():
            return """
### Connection Issue

I'm experiencing connectivity issues. Please check your API key configuration.

**Troubleshooting Steps:**
1. Verify your API key is valid
2. Check your internet connection
3. Try again in a few moments
            """
        
        # Create conversation context with system prompt and past messages
        conversation = [SYSTEM_PROMPT]
        
        # Add memory context if available
        for msg in memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                conversation.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                conversation.append(f"Assistant: {msg.content}")
        
        # Add structured context prompt if available
        if context:
            context_prompt = format_structured_context(context)
            conversation.append(f"Context: {context_prompt}")
        
        # Add explicit instruction for structured response
        structured_instruction = """
Remember to format your response with clear structure using:
- Markdown headings (### for sections)
- Tables for comparing information
- Bullet points or numbered lists for steps
- Bold text for important points
- Keep it concise and scannable
        """
        conversation.append(structured_instruction)
        
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
        
        return """
### Response Generation Issue

I couldn't generate a response at this time.

**Please try:**
- Rephrasing your question
- Breaking complex questions into simpler ones
- Trying again in a moment
        """
        
    except Exception as e:
        print(f"Error in direct_generate_response: {str(e)}")
        return f"""
### Technical Difficulty

I encountered an issue with the response generation.

**Error:** {str(e)[:50]}...

Please try again with a different question or approach.
        """

def format_structured_context(context: Dict[str, Any]) -> str:
    """
    Format context information in a structured way for better LLM prompting
    
    Args:
        context: Context dictionary with information
        
    Returns:
        str: Formatted context string
    """
    if context.get("type") == "job_listings" and context.get("jobs"):
        jobs = context.get("jobs", [])
        job_table = "Previous jobs we discussed:\n\n"
        job_table += "| Title | Company | Location | Type | Skills |\n"
        job_table += "| ----- | ------- | -------- | ---- | ------ |\n"
        
        for job in jobs[:3]:  # Limit to first 3 jobs
            title = job.get('title', 'Unknown')
            company = job.get('company', 'Unknown')
            location = job.get('location', 'Unknown')
            job_type = job.get('type', 'Unknown')
            skills = ', '.join(job.get('skills', ['Unknown'])[:3])
            
            job_table += f"| {title} | {company} | {location} | {job_type} | {skills} |\n"
            
        return job_table
    
    # Generic context formatting for other context types
    return json.dumps(context, indent=2)

def generate_career_advice(query: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate structured career advice using LangChain with fallback options
    
    Args:
        query (str): User's question
        context (dict, optional): Additional context about previous interactions
    
    Returns:
        str: Generated structured response or redirection message if off-topic
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
                print("User expressing job interest, generating structured job details response")
                return generate_structured_job_response(query, context.get("jobs", []))
        
        # For other queries, check if it's a career question
        if not is_career_question(query, context):
            print("Not a career question")
            return """
### Career Focus Reminder

I'm designed to assist with career-related questions including:

- **Job searching** and applications
- **Resume building** and optimization
- **Interview preparation** and techniques
- **Professional development** strategies
- **Workplace dynamics** and advancement
- **Mentorship** and networking

Could you please ask something related to these areas so I can help with your professional journey?
            """
        
        print("Career question detected, continuing...")
        
        # Process context if available
        if context:
            print(f"Injecting context: {context.get('type', 'unknown')}")
            inject_context_into_memory(context)
        
        # Enhanced structure prompt
        structure_instruction = """
Format your response in a highly structured manner:
- Use markdown headings (###) for clear sections
- Use bullet points or numbered lists for steps/points
- Create tables for comparative information
- Bold important concepts
- IMPORTANT: When mentioning courses, websites or resources, ALWAYS use Markdown hyperlinks like [Course Name](URL)
- Keep your response concise and easy to scan
    """
        
        print("Getting conversation chain...")
        # Try LangChain implementation first
        chain = get_conversation_chain()
        
        if chain:
            try:
                # Add user query to memory
                memory.chat_memory.add_user_message(query)
                
                # Generate response using the chain with structure instruction
                print("Generating response with LangChain...")
                enhanced_query = f"{query}\n\n{structure_instruction}"
                response = chain.predict(input=enhanced_query)
                
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
            return """
### Technical Difficulty

I'm experiencing technical difficulties. Here's some general career advice:

**Key Career Success Factors:**
- **Networking:** Build meaningful professional relationships
- **Continuous learning:** Stay updated with industry trends
- **Personal branding:** Develop your unique professional identity
- **Career planning:** Set clear goals with actionable steps
- **Work-life balance:** Maintain sustainable productivity

Need specific advice? Please try asking again with a focused question.
            """

def inject_context_into_memory(context: Dict[str, Any]) -> None:
    """
    Inject contextual information into the conversation memory
    
    Args:
        context (dict): Context information to inject
    """
    try:
        if context.get("type") == "job_listings" and context.get("jobs"):
            # Format job info in structured format
            jobs = context.get("jobs", [])
            job_titles = [f"{job.get('title', 'a job')} at {job.get('company', 'a company')}" for job in jobs]
            
            job_context_msg = f"""
### Jobs Currently Discussing

We're currently discussing these opportunities:
- {job_titles[0] if len(job_titles) > 0 else 'N/A'}
- {job_titles[1] if len(job_titles) > 1 else 'N/A'}
- {job_titles[2] if len(job_titles) > 2 else 'N/A'}
            """
            
            # Add as AI message to memory
            memory.chat_memory.add_ai_message(job_context_msg)
            print(f"Added structured context to memory")
            
            # Also add user preferences if available
            if "user_preferences" in context:
                pref_msg = f"""
### Your Job Preferences
                
You've mentioned interest in: **{context.get('user_preferences')}**
                
Let me tailor my recommendations to these preferences.
                """
                memory.chat_memory.add_ai_message(pref_msg)
                print(f"Added structured preferences to memory")
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
    is_short = len(query.split()) <= 8
    has_followup_phrase = any(phrase in query_lower for phrase in followup_phrases)
    
    # More permissive criteria - either short or contains follow-up phrase
    return is_short or has_followup_phrase

def generate_structured_job_response(query: str, jobs: List[Dict[str, Any]]) -> str:
    """
    Generate a detailed, structured response about jobs with tables and formatted sections
    
    Args:
        query (str): User's follow-up question
        jobs (list): List of job details
    
    Returns:
        str: Formatted response with job details in an organized, tabular format
    """
    print(f"Generating structured job details for: {query}")
    print(f"Jobs available: {len(jobs)}")
    
    if not jobs:
        return """
### Job Search Assistance

I don't have any job details to share at the moment.

**Would you like me to help you:**
- Search for specific job opportunities?
- Discuss job search strategies?
- Review your resume for job applications?
- Practice interview questions?

Let me know what interests you most!
        """
    
    # Try to identify which specific job the user is asking about
    query_lower = query.lower()
    selected_job = None
    
    # Enhanced matching algorithm for job selection
    for job in jobs:
        title = job.get('title', '').lower()
        company = job.get('company', '').lower()
        
        # Multiple pattern matching approaches for flexibility
        if (title in query_lower and company in query_lower) or \
           (f"{title} at {company}" in query_lower) or \
           (title in query_lower and any(company_part in query_lower for company_part in company.split())) or \
           (any(word in query_lower for word in title.split()) and company in query_lower):
            selected_job = job
            print(f"Found specific job match: {title} at {company}")
            break
    
    # Handle affirmative responses
    if not selected_job and (
        query_lower in ["yes", "sure", "okay", "ok", "please", "definitely", "absolutely"] or
        any(affirmation in query_lower for affirmation in ["yes", "want", "like to", "interested", "tell me more"])
    ):
        selected_job = jobs[0]
        print(f"User affirmed interest - using first job: {selected_job.get('title', 'Unknown job')}")
    
    # Default to first job if no match found
    if not selected_job and jobs:
        selected_job = jobs[0]
        print(f"No specific match, using first job by default: {selected_job.get('title', 'Unknown job')}")
    
    # Extract all job details
    job_title = selected_job.get('title', 'Job Position')
    job_company = selected_job.get('company', 'Company')
    job_location = selected_job.get('location', 'Location not specified')
    job_type = selected_job.get('type', 'Type not specified')
    
    # Format skills properly
    job_skills_raw = selected_job.get('skills', [])
    if isinstance(job_skills_raw, list):
        job_skills = ", ".join(job_skills_raw)
    else:
        job_skills = str(job_skills_raw)
    
    # Check for different question types to provide structured responses
    # Application tips request
    if any(keyword in query_lower for keyword in ["apply", "application", "resume", "tips", "advice", "how to"]):
        return f"""
### Application Guide: {job_title} at {job_company}

| **Detail** | **Information** |
|------------|-----------------|
| **Position** | {job_title} |
| **Company** | {job_company} |
| **Location** | {job_location} |
| **Job Type** | {job_type} |

#### Required Skills
- {job_skills.replace(', ', '\n- ')}

#### Application Strategy

| **Step** | **Action Items** |
|----------|------------------|
| **1. Research** | • Study {job_company}'s products/services<br>• Understand their mission and values<br>• Research recent company news |
| **2. Resume** | • Highlight experience with key skills<br>• Quantify achievements with metrics<br>• Use job description keywords |
| **3. Cover Letter** | • Address specific job requirements<br>• Show enthusiasm for {job_company}<br>• Connect your experience to their needs |
| **4. Interview** | • Prepare {job_title}-specific examples<br>• Research typical technical questions<br>• Prepare thoughtful questions about the role |
| **5. Follow-up** | • Send thank-you email within 24 hours<br>• Reference specific conversation points<br>• Restate your interest and qualifications |

**Would you like specific interview questions to prepare for this role?**
"""

    # Salary/compensation question
    elif any(keyword in query_lower for keyword in ["salary", "pay", "compensation", "money", "benefit"]):
        return f"""
### Compensation Information: {job_title} at {job_company}

| **Detail** | **Information** |
|------------|-----------------|
| **Position** | {job_title} |
| **Company** | {job_company} |
| **Location** | {job_location} |

#### Typical Compensation Range

Based on industry standards for {job_title} roles at companies like {job_company} in {job_location}:

| **Component** | **Typical Range** |
|---------------|-------------------|
| **Base Salary** | Typically $85,000-$120,000 for this role |
| **Bonuses** | Performance bonuses of 5-15% annual salary |
| **Stock Options** | Often included at tech companies like {job_company} |
| **Benefits** | Health insurance, retirement plans, and paid time off |

**Note:** Actual compensation may vary based on experience level, specific skills (particularly {job_skills.split(',')[0]}), and negotiation.

#### Negotiation Tips
1. **Research thoroughly** - Use Glassdoor, LinkedIn Salary, and PayScale
2. **Highlight your expertise** in {job_skills.split(',')[0]} and {job_skills.split(',')[1] if ',' in job_skills else 'related technologies'}
3. **Consider the total package** - not just base salary

Would you like specific negotiation strategies for {job_company}?
"""

    # Skills/requirements question
    elif any(keyword in query_lower for keyword in ["skill", "requirement", "qualification", "need", "expect"]):
        return f"""
### Skills & Requirements: {job_title} at {job_company}

| **Core Detail** | **Information** |
|-----------------|-----------------|
| **Position** | {job_title} |
| **Company** | {job_company} |
| **Type** | {job_type} |

#### Technical Skills Required

#### Skill Breakdown

| **Skill Category** | **Importance** | **Details** |
|-------------------|----------------|------------|
| **Technical Skills** | ★★★★★ | Proficiency in {', '.join(job_skills_raw[:3] if isinstance(job_skills_raw, list) and len(job_skills_raw) >= 3 else ['relevant technologies'])} |
| **Problem Solving** | ★★★★☆ | Ability to troubleshoot complex systems |
| **Communication** | ★★★★☆ | Clear documentation and team collaboration |
| **Project Management** | ★★★☆☆ | Handling deadlines and resource planning |

#### Qualification Levels

- **Required:** Bachelor's degree in Computer Science or related field
- **Preferred:** 3+ years experience with {job_skills.split(',')[0] if ',' in job_skills else 'relevant technologies'}
- **Bonus:** Experience with {job_company}-specific tools or methodologies

**Do you have specific questions about any of these requirements?**
"""

    # Company/culture question
    elif any(keyword in query_lower for keyword in ["company", "culture", "work", "environment", "team", "about"]):
        return f"""
### Company Profile: {job_company}

| **Company Detail** | **Information** |
|-------------------|-----------------|
| **Industry** | Technology/Software |
| **Size** | Medium to Large Enterprise |
| **Culture** | Innovation-focused, collaborative |

#### Work Environment

| **Aspect** | **Description** |
|------------|-----------------|
| **Work Style** | {job_type} with flexible scheduling |
| **Team Structure** | Cross-functional teams with agile methodology |
| **Career Growth** | Internal promotion pathways and learning opportunities |
| **Work-Life Balance** | Competitive PTO and wellness programs |

#### Company Values
- **Innovation:** Pushing boundaries in technology
- **Collaboration:** Team-based problem solving
- **Excellence:** High standards for quality
- **Diversity:** Inclusive workplace policies

#### Position Details
This {job_title} role focuses on delivering high-quality solutions while working with a diverse team of professionals. The role offers exposure to cutting-edge technologies and challenging projects.

**Would you like to know more about the team structure or advancement opportunities?**
"""

    # Interview preparation question
    elif any(keyword in query_lower for keyword in ["interview", "question", "prepare", "ask"]):
        return f"""
### Interview Preparation: {job_title} at {job_company}

| **Position** | **Location** | **Type** |
|--------------|--------------|----------|
| {job_title} | {job_location} | {job_type} |

#### Common Interview Questions

| **Question Type** | **Example Questions** |
|-------------------|----------------------|
| **Technical** | • How would you implement [relevant feature] using {job_skills.split(',')[0]}?<br>• Explain your approach to debugging in {job_skills.split(',')[1] if ',' in job_skills else 'your preferred language'}<br>• How would you optimize a slow-performing application? |
| **Behavioral** | • Describe a challenging project and how you overcame obstacles<br>• How do you handle tight deadlines and changing requirements?<br>• Tell me about a time you resolved a conflict in a team |
| **Problem-Solving** | • How would you design a system for [relevant task]?<br>• Walk through your thought process on [domain-specific problem]<br>• What metrics would you use to measure success? |

#### Interview Tips

1. **Prepare examples** that highlight your experience with {job_skills.split(',')[0]}
2. **Research {job_company}** - understand their products, culture, recent news
3. **Practice explaining** complex technical concepts clearly
4. **Prepare questions** about the team, projects, and growth opportunities

**Would you like mock interview questions specific to this role?**
"""

    # Default comprehensive job overview
    else:
        return f"""
### {job_title} at {job_company}

| **Job Detail** | **Information** |
|---------------|-----------------|
| **Company** | {job_company} |
| **Location** | {job_location} |
| **Job Type** | {job_type} |

#### Skills Required

#### Role Overview
This {job_title} position focuses on designing, implementing, and maintaining systems that align with {job_company}'s business objectives. The role requires strong expertise in {job_skills.split(',')[0]} and collaboration with cross-functional teams.

#### Key Responsibilities

| **Area** | **Tasks** |
|----------|-----------|
| **Development** | • Build robust, scalable solutions<br>• Optimize performance<br>• Implement security best practices |
| **Collaboration** | • Work with product and design teams<br>• Participate in code reviews<br>• Mentor junior team members |
| **Innovation** | • Research new technologies<br>• Propose improvements<br>• Contribute to architectural decisions |

#### Career Path
This role provides growth opportunities toward Senior {job_title}, Team Lead, and eventually Technical Management or Architecture roles.

**What specific aspect of this position would you like to explore further?**
"""
    

def generate_course_recommendations(query: str) -> str:
    """Generate course recommendations with proper hyperlinks based on user query"""
    
    # Detect tech/course interests in the query
    query_lower = query.lower()
    
    # Format the prompt to ensure links are included
    prompt = f"""
You are a career advisor specializing in tech education recommendations.
The user is asking about courses: "{query}"

Provide a comprehensive answer with the following structure:
1. A brief TL;DR summary section at the top
2. An explanation of the learning path for this technology
3. A detailed table of FREE course recommendations with the following columns:
   - Course Provider (e.g., Coursera, edX, freeCodeCamp)
   - Course Name (with WORKING hyperlink to the actual course)
   - Focus/Topics
   - Key Features (what makes this course valuable)

IMPORTANT: 
- ALL LINKS MUST BE FUNCTIONAL AND PROPERLY FORMATTED AS MARKDOWN: [Course Name](https://example.com)
- Only include FREE courses with direct links
- Include at least 5 different course options
- Format the response with markdown for readability
"""
    
    # Use your existing function to generate the response
    try:
        # Create a simple context object
        context = {
            "type": "course_recommendation",
            "area": "tech education",
            "focus": "providing structured course recommendations with hyperlinks"
        }
        
        # Call your existing function or API
        return direct_generate_response(prompt, context)
    except Exception as e:
        print(f"Error generating course recommendations: {str(e)}")
        # Fallback response with links
        return """
## Free Java Frontend Development Courses

Since Java is primarily used for backend development, for frontend you'll need to learn:

1. **HTML/CSS/JavaScript** - Frontend fundamentals
2. **A JavaScript framework** - Like React, Angular or Vue.js
3. **Connecting frontend to Java backend**

### Recommended Free Courses:

* [freeCodeCamp - Responsive Web Design](https://www.freecodecamp.org/learn/responsive-web-design/) - Learn HTML/CSS fundamentals
* [freeCodeCamp - JavaScript Algorithms](https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/) - JavaScript basics
* [Codecademy - Learn Java](https://www.codecademy.com/learn/learn-java) - Java fundamentals
* [MDN Web Docs](https://developer.mozilla.org/en-US/docs/Learn) - Comprehensive web development tutorials
* [Java Brains YouTube Channel](https://www.youtube.com/c/JavaBrainsChannel) - Java and frontend tutorials

Would you like me to recommend more specific courses based on your experience level?
"""

def is_career_question(query: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Determine if a query is career-related using a comprehensive topic classification approach
    
    Args:
        query (str): User query
        context (dict, optional): Additional context from previous messages
    
    Returns:
        bool: True if career-related, False otherwise
    """
    # Always consider job context as career related
    if context and context.get("type") == "job_listings":
        return True
    
    # Check for job interest phrases that might not be caught by keywords
    job_interest_phrases = [
        "interested in", "tell me more about", "more about the", "like to know more",
        "want to apply", "how to apply", "yes", "please", "sure", "sounds good",
        "like the", "would like", "about this job", "about that job", "about the position",
        "want to learn", "job description", "requirements", "qualifications", "salary",
        "sounds interesting", "company", "location", "remote", "hybrid", "onsite",
        "prepare for", "apply for", "perfect", "good fit", "skills needed" 
        "want to apply", "how to apply", "yes", "please", "sure", "sounds good"
    ]
    
    query_lower = query.lower()
    if any(phrase in query_lower for phrase in job_interest_phrases):
        return True
    
    # Allow follow-up questions in an ongoing conversation
    if is_followup_question(query) and memory.chat_memory.messages:
        return True
    
    # Define relevant topics and their keywords (comprehensive list)
    relevant_topics = {
        "career_development": [
            "career", "profession", "professional", "growth", "advance", "progress", "path", 
            "trajectory", "promotion", "transition", "pivot", "change", "switch", "move", 
            "advancement", "aspiration", "goal", "objective", "ambition", "direction", "success",
            "industry", "field", "sector", "domain", "area", "discipline", "specialty"
            "trajectory", "promotion", "transition", "pivot", "change", "switch"
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
            "application", "apply", "workplace", "company", "organization", "remote", "hybrid"
        ],
        "resume_portfolio": [
            "experience", "background", "history", "track record", "qualification", "credential",
            "achievement", "accomplishment", "education", "degree", "diploma", "certification",
            "certificate", "training", "skill", "competency", "ability", "capability", "expertise",
            "proficiency", "mastery", "strength", "talent", "aptitude", "flair", "document"
            "resume", "cv", "curriculum vitae", "bio", "portfolio", "profile", 
            "experience", "background", "history", "qualification", "credential"
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
            "interview", "recruiter", "hiring manager", "question", "answer", "preparation", 
            "behavioral", "technical", "screening", "assessment", "evaluation"
        ]
    }
    
    # Check if the query contains any relevant keywords
    for topic, keywords in relevant_topics.items():
        if any(keyword in query_lower for keyword in keywords):
            return True
    
    # Additional check for general career phrases
    general_career_phrases = [
        "how to get a", "how to become", "what should i do", "i need help with my", 
        "looking for advice", "need guidance", "struggling with", "tips for"
    ]
    
    if any(phrase in query_lower for phrase in general_career_phrases):
        return True
    
    return False

def reset_conversation():
    """Clear the conversation memory"""
    global memory
    memory = ConversationBufferMemory(return_messages=True)

    