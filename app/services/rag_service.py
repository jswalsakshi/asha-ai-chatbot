import faiss
import numpy as np
from dotenv import load_dotenv
load_dotenv()

import os
import requests
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict, Any, Optional

HUGGINGFACE_API_URL = os.environ["HUGGINGFACE_API_URL"]
HUGGINGFACE_API_TOKEN = os.environ["HUGGINGFACE_API_TOKEN"]

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")  # Where you cloned the models
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "utils", "models", "bge-small-en-v1.5")
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- LOAD MODELS ---
# 1. Load embedding model (local)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

def generate_response_huggingface(query: str, context: str) -> str:
    """
    Generate a structured response using Hugging Face API
    
    Args:
        query: User query
        context: Context information
        
    Returns:
        str: Generated response
    """
    # Enhance prompt to encourage structured, tabular responses
    structured_prompt = f"""
    Question: {query}
    
    Context: {context}
    
    Instructions:
    - Format your response using Markdown
    - Use tables (| header | header |) for structured information
    - Use headings (###) for sections
    - Use bullet points or numbered lists for steps
    - Bold important information
    - Keep response concise and well-organized
    
    Answer:
    """
    
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
    }
    payload = {
        "inputs": structured_prompt
    }
    
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload, verify=False)
    response_json = response.json()
    
    # Assuming the response contains a "generated_text" field
    return response_json[0]['generated_text']

# Absolute path to the local model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "utils", "models", "bge-small-en-v1.5")

# Paths to index and chunks
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(DATA_DIR, "job_chunks.npy")

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

# Load text chunks (job listings)
job_chunks = np.load(CHUNKS_PATH, allow_pickle=True)

# Load local embedding model
model = SentenceTransformer(MODEL_PATH)

def semantic_search(query: str, top_k: int = 3) -> List[str]:
    """
    Returns top_k most semantically similar job chunks for the user query.
    Enhanced to handle more query types and provide better structured results.
    
    Args:
        query: User query
        top_k: Number of results to return
        
    Returns:
        List[str]: List of relevant job chunks
    """
    # Enhance query for better job matching
    enhanced_query = enhance_job_query(query)
    
    # Generate embeddings
    query_vector = model.encode([enhanced_query])
    
    # Search FAISS index
    indices = index.search(np.array(query_vector), top_k)[1]
    
    # Get results
    results = [job_chunks[i] for i in indices[0]]
    
    # Format results in a more structured way
    return results

def enhance_job_query(query: str) -> str:
    """
    Enhance job query for better semantic search results
    
    Args:
        query: Original user query
        
    Returns:
        str: Enhanced query
    """
    query_lower = query.lower()
    
    # Generic affirmative responses
    if query_lower in ["yes", "sure", "ok", "okay", "tell me more", "go ahead", "please"]:
        return "show me detailed information about available jobs"
    
    # Extract industry or role information
    if "industry" in query_lower or "sector" in query_lower:
        return f"jobs in {query_lower} industry sector"
        
    # Specific job role interest
    job_roles = ["developer", "engineer", "manager", "designer", "analyst", 
                "scientist", "consultant", "specialist", "director"]
    
    for role in job_roles:
        if role in query_lower:
            return f"job openings for {role} positions"
            
    # Specific skills interest
    tech_skills = ["python", "java", "javascript", "react", "node", "angular",
                  "machine learning", "data science", "cloud", "aws", "azure"]
                  
    for skill in tech_skills:
        if skill in query_lower:
            return f"jobs requiring {skill} skills"
    
    # Default enhancement
    return f"{query} jobs and career opportunities"

def format_job_response(jobs: List[Dict[str, Any]], query: str = "") -> str:
    """
    Format job listings into a structured, tabular markdown response
    
    Args:
        jobs: List of job dictionaries
        query: The original user query
        
    Returns:
        str: Formatted markdown response
    """
    if not jobs:
        return ("### No Matching Jobs Found\n\n"
                "I couldn't find jobs matching your criteria. Would you like to:\n"
                "1. Try a different search term?\n"
                "2. Browse all available positions?\n"
                "3. Discuss what skills might enhance your job prospects?")
    
    # Structure as markdown table
    response = "### Job Opportunities\n\n"
    response += "| Title | Company | Location | Type | Key Skills |\n"
    response += "|-------|---------|----------|------|------------|\n"
    
    for job in jobs[:5]:  # Limit to 5 jobs
        title = job.get("title", "Untitled")
        company = job.get("company", "Unknown")
        location = job.get("location", "Unspecified")
        job_type = job.get("type", "Full-time")
        
        # Format skills as comma-separated list
        skills_raw = job.get("skills", [])
        if isinstance(skills_raw, list):
            skills = ", ".join(skills_raw[:3])  # Top 3 skills
            if len(skills_raw) > 3:
                skills += "..."
        else:
            skills = str(skills_raw)
            
        response += f"| **{title}** | {company} | {location} | {job_type} | {skills} |\n"
    
    # Add helpful prompt
    response += "\n### Would you like to:\n"
    response += "1. See details about a specific job? (just mention the job title or company)\n"
    response += "2. Get application tips for these positions?\n"
    response += "3. Search for different roles or industries?"
    
    return response

def get_detailed_job_info(job: Dict[str, Any]) -> str:
    """
    Generate detailed information about a specific job in a structured format
    
    Args:
        job: Job dictionary
        
    Returns:
        str: Formatted job details
    """
    title = job.get("title", "Job Position")
    company = job.get("company", "Company")
    location = job.get("location", "Location not specified")
    job_type = job.get("type", "Full-time")
    
    # Format skills
    skills_raw = job.get("skills", [])
    if isinstance(skills_raw, list):
        skills = ", ".join(skills_raw)
    else:
        skills = str(skills_raw)
    
    # Create structured response
    response = f"### {title} at {company}\n\n"
    
    # Job details table
    response += "| **Detail** | **Information** |\n"
    response += "|------------|----------------|\n"
    response += f"| **Position** | {title} |\n"
    response += f"| **Company** | {company} |\n"
    response += f"| **Location** | {location} |\n"
    response += f"| **Job Type** | {job_type} |\n\n"
    
    # Skills section
    response += "### Required Skills\n\n"
    if isinstance(skills_raw, list):
        for skill in skills_raw:
            response += f"- {skill}\n"
    else:
        response += f"- {skills}\n"
    
    # Job description
    description = job.get("description", "")
    if description:
        response += f"\n### Job Description\n\n{description}\n"
    else:
        response += "\n### Role Overview\n\n"
        response += f"This {title} role at {company} involves developing and maintaining "
        response += f"software systems, collaborating with cross-functional teams, and "
        response += f"contributing to the company's technical growth.\n"
    
    # Application tips
    response += "\n### Application Tips\n\n"
    response += "1. **Highlight relevant skills** - Especially focus on your experience with " 
    response += f"{', '.join(skills_raw[:2]) if isinstance(skills_raw, list) and skills_raw else 'relevant technologies'}\n"
    response += "2. **Research the company** - Understand " + company + "'s products, culture and recent news\n"
    response += "3. **Prepare examples** - Be ready to discuss specific projects where you've used relevant skills\n"
    response += "4. **Ask thoughtful questions** - Demonstrate your interest in the role and company\n"
    
    return response

# --- TEST ---
if __name__ == "__main__":
    query = "Are there any remote frontend jobs?"
    context = "\n".join(semantic_search(query))
    print("Context:", context)
    print("Response from Hugging Face API:", generate_response_huggingface(query, context))