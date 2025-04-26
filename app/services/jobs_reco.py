import pandas as pd
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_jobs_from_csv():
    """Load job listings from CSV file and organize by category"""
    try:
        # Find the CSV file path
        # Look in a few common locations relative to this file
        base_dir = Path(__file__).parent.parent.parent  # Go up to project root
        possible_paths = [
            base_dir / "data" / "job_listing_data.csv",
            base_dir / "frontend" / "data" / "job_listing_data.csv",
            base_dir / "job_listing_data.csv"
        ]
        
        csv_path = None
        for path in possible_paths:
            if path.exists():
                csv_path = str(path)
                break
                
        if not csv_path:
            logger.warning(f"Job data file not found in any of the expected locations")
            return {}
            
        # Load CSV into DataFrame
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} jobs from {csv_path}")
        
        # Categorize jobs based on skills/title
        job_categories = {}
        
        # Define category keywords
        keywords = {
            "tech": ["python", "java", "developer", "engineer", "data", "machine learning", 
                    "ai", "backend", "frontend", "software", "coding", "programming", "IT"],
            "marketing": ["marketing", "digital", "content", "seo", "social media", 
                         "brand", "communications", "PR", "advertising"],
            "finance": ["finance", "accounting", "banking", "financial", "analyst", 
                       "investment", "economics", "budget", "trading", "audit"],
            "hr": ["hr", "human resources", "recruiting", "talent", "hiring", 
                  "people operations", "training", "personnel"]
        }
        
        # Process each job
        for _, job in df.iterrows():
            # Convert skills to a list if it's a string
            if isinstance(job["skills"], str):
                skills_list = [skill.strip() for skill in job["skills"].split(",")]
            else:
                skills_list = []
            
            # Determine category based on job title and skills
            job_text = (job["job_title"] + " " + " ".join(skills_list)).lower()
            assigned_category = "other"
            
            for category, category_keywords in keywords.items():
                if any(keyword in job_text for keyword in category_keywords):
                    assigned_category = category
                    break
            
            # Create job dict with proper error handling for missing columns
            job_dict = {
                "title": job["job_title"],
                "company": job["company"],
                "location": job.get("location", "Location not specified"),
                "job_type": job.get("job_type", "Full-time"),
                "skills": skills_list,
                "apply_link": job.get("apply_link", "#"),
                "category": assigned_category  # Store the category with the job
            }
            
            # Add to appropriate category
            if assigned_category not in job_categories:
                job_categories[assigned_category] = []
            
            job_categories[assigned_category].append(job_dict)
        
        # Ensure all categories exist even if empty
        for category in keywords.keys():
            if category not in job_categories:
                job_categories[category] = []
                
        return job_categories
        
    except Exception as e:
        logger.error(f"Error loading jobs: {str(e)}")
        return {}

def extract_job_category(user_messages):
    """Extract job category based on user's chat history"""
    # Join all user messages into a single string for analysis
    if isinstance(user_messages, list):
        all_messages = " ".join([
            msg.get("content", "") for msg in user_messages 
            if isinstance(msg, dict) and msg.get("role") == "user"
        ])
    else:
        all_messages = str(user_messages)
        
    all_messages = all_messages.lower()
    
    # Simple keyword matching for categories
    keywords = {
        "tech": ["software", "developer", "coding", "programming", "engineer", "tech", "data", "IT", 
                 "python", "java", "web", "app", "devops", "cloud", "AI", "machine learning"],
        "marketing": ["marketing", "brand", "content", "social media", "SEO", "digital", "campaign", 
                      "advertising", "communications", "PR"],
        "finance": ["finance", "accounting", "investment", "banking", "financial", "analyst", 
                    "economics", "budget", "trading", "audit"],
        "hr": ["HR", "human resources", "recruiting", "talent", "hiring", "people operations", 
               "training", "personnel", "employees", "recruitment"]
    }
    
    # Count keyword matches for each category
    scores = {category: 0 for category in keywords}
    for category, words in keywords.items():
        for word in words:
            if word in all_messages:
                scores[category] += 1
    
    # Return the category with the highest score, or "tech" as default if no matches
    if max(scores.values(), default=0) > 0:
        return max(scores.items(), key=lambda x: x[1])[0]
    return "tech"  # Default category

def get_recommended_jobs(user_messages, job_database=None, num_jobs=3):
    """Get personalized job recommendations based on user messages"""
    # Load jobs if not provided
    if job_database is None:
        job_database = load_jobs_from_csv()
    
    # Extract relevant category
    category = extract_job_category(user_messages)
    logger.info(f"Extracted job category: {category} for user query")
    
    # Get jobs from the determined category
    jobs = job_database.get(category, [])
    
    # If not enough jobs in the category, get from other categories
    if len(jobs) < num_jobs:
        # Collect all jobs from all categories
        all_jobs = []
        for cat, cat_jobs in job_database.items():
            if cat != category:  # Don't duplicate from the main category
                all_jobs.extend(cat_jobs)
        
        # Add enough jobs from other categories to reach num_jobs
        remaining_slots = num_jobs - len(jobs)
        if all_jobs and remaining_slots > 0:
            jobs.extend(all_jobs[:remaining_slots])
    
    # Take top N jobs
    return jobs[:num_jobs]

def format_job_listings(jobs):
    """Format job listings for display in chat"""
    if not jobs:
        return "No matching jobs found at this time. Please try adjusting your search criteria."
        
    result = "Here are some job opportunities that might interest you:\n\n"
    for i, job in enumerate(jobs, 1):
        result += f"**{i}. {job['title']} at {job['company']}**\n"
        result += f"📍 {job['location']} | 💼 {job.get('job_type', 'Full-time')}\n"
        result += f"🔍 Skills: {', '.join(job['skills'])}\n"
        
        # Add application link if available
        apply_link = job.get('apply_link', '')
        if apply_link and apply_link != '#':
            result += f"🔗 [Apply Now]({apply_link})\n"
        
        result += "\n"
    
    result += "Would you like more details about any of these positions?"
    return result

def search_jobs(query, job_database=None, max_results=5):
    """Search for jobs based on a query string"""
    if job_database is None:
        job_database = load_jobs_from_csv()
        
    query = query.lower()
    matches = []
    
    # Flatten jobs from all categories
    all_jobs = []
    for category_jobs in job_database.values():
        all_jobs.extend(category_jobs)
    
    # Search in all jobs
    for job in all_jobs:
        # Create searchable text from job details
        searchable_text = f"{job['title']} {job['company']} {job['location']} {' '.join(job['skills'])}".lower()
        
        # Check if query terms appear in the searchable text
        if query in searchable_text:
            matches.append(job)
            
    # Return up to max_results matches
    return matches[:max_results]

# Test the module if run directly
if __name__ == "__main__":
    print("Testing job recommendation module...")
    
    # Test loading jobs
    jobs = load_jobs_from_csv()
    print(f"Loaded {sum(len(jobs[cat]) for cat in jobs)} jobs across {len(jobs)} categories")
    
    # Test recommendation
    test_messages = [
        {"role": "user", "content": "I'm looking for software engineering positions"}
    ]
    recommendations = get_recommended_jobs(test_messages, jobs)
    print("\nRecommendations:")
    for job in recommendations:
        print(f"- {job['title']} at {job['company']}")
    
    # Test formatting
    print("\nFormatted output:")
    print(format_job_listings(recommendations))