import streamlit as st
import pandas as pd
import rag_service 

# --- Configuration ---
JOB_DATA_FILE = "job_listing_data.csv"

# --- Data Loading (Cached) ---
@st.cache_data
def load_job_data(file_path):
    """Loads job listing data from the specified CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Add a unique ID column if it doesn't exist (using index)
        if 'job_id' not in df.columns:
             df['job_id'] = df.index # Simple way to add an ID
        df = df.fillna('N/A')
        for col in ['Job Title', 'Company Name', 'Location', 'Description', 'Skills Required']: # Adjust as needed
             if col in df.columns:
                 df[col] = df[col].astype(str)
        return df
    except FileNotFoundError:
        st.error(f"Error: Job data file not found at '{file_path}'.")
        return None
    except Exception as e:
        st.error(f"Error loading job data: {e}")
        return None

# --- Job Display Formatting (Keep as is) ---
def format_job_display(job_series):
    # ... (your existing formatting code) ...
    title = job_series.get('Job Title', 'N/A')
    company = job_series.get('Company Name', 'N/A')
    location = job_series.get('Location', 'N/A')
    description = job_series.get('Description', 'No description available.')
    skills = job_series.get('Skills Required', 'N/A')
    apply_link = job_series.get('Apply Link', '#')

    with st.expander(f"{title} at {company} ({location})", expanded=False):
        st.markdown(f"**Skills Required:** {skills}")
        st.markdown(f"**Description:**\n{description}")
        if apply_link != '#':
            st.link_button("Apply Here", apply_link)
        else:
            st.caption("No direct apply link provided.")

# --- Main Job Feature Function ---
def display_job_feature():
    """Handles the display and interaction for the Jobs context."""

    # Load the main structured job data
    df_jobs = load_job_data(JOB_DATA_FILE)

    if df_jobs is None or df_jobs.empty:
        st.warning("No job data is currently available.")
        return

    # Ensure RAG utils are ready (models loaded etc.) - optional check
    if rag_service.embedding_model is None or rag_service.index is None:
         st.warning("Job search may be limited as RAG components failed to load.")
         # Optionally fall back to keyword search here if needed

    st.write("How can I help you with jobs today?")

    # --- Interaction Options (Keep Show All, Modify Search) ---
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Show All Recent Jobs", key="jobs_show_all", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Show me all recent jobs."})
            st.session_state.action = "show_all_jobs"
            st.rerun() # Rerun to handle the action

    with col2:
         if st.button("Search Specific Jobs", key="jobs_search", use_container_width=True):
             st.session_state.messages.append({"role": "user", "content": "I want to search for specific jobs."})
             st.session_state.action = "prompt_search" # Set an action flag
             st.rerun() # Rerun to show the input prompt


    # --- Handle Actions based on Flags ---

    # 1. Show All Jobs Action (Keep as is)
    if st.session_state.get("action") == "show_all_jobs":
        st.write("Here are some of the available jobs:")
        num_jobs_to_show = min(10, len(df_jobs))
        bot_message_content = f"Okay, here are the {num_jobs_to_show} most recent job listings I have:"
        st.session_state.messages.append({"role": "assistant", "content": bot_message_content})
        # Display directly
        for index, job in df_jobs.head(num_jobs_to_show).iterrows():
            format_job_display(job) # Use the existing formatter
        st.session_state.action = None # Reset action
        st.rerun()


    # 2. Prompt for Search Term Action (Keep as is)
    if st.session_state.get("action") == "prompt_search":
        search_query = st.text_input("What kind of job are you looking for? (e.g., 'remote python developer', 'marketing manager in Mumbai')", key="job_search_query")
        if st.button("Find Jobs", key="find_jobs_button"):
            if search_query:
                st.session_state.messages.append({"role": "user", "content": f"Search jobs for: {search_query}"})
                st.session_state.search_query = search_query
                st.session_state.action = "perform_semantic_search" # Change action name
                st.rerun()
            else:
                st.warning("Please enter what you're looking for.")


    # 3. Perform SEMANTIC Search Action (Modified)
    if st.session_state.get("action") == "perform_semantic_search":
        query = st.session_state.get("search_query", "")
        if query:
            with st.spinner(f"Searching for '{query}'..."):
                # --- Use RAG Semantic Search ---
                retrieved_ids = rag_service.semantic_search_job_ids(query, top_k=10) # Get top 10 relevant IDs
                # --- End RAG Search ---

                if retrieved_ids:
                    # --- Filter the main DataFrame using the retrieved IDs ---
                    # Ensure the ID column used here matches the one stored in chunks ('job_id')
                    results_df = df_jobs[df_jobs['job_id'].isin(retrieved_ids)].copy()
                    # Optional: Re-order results based on retrieval order?
                    # results_df['sort_order'] = results_df['job_id'].map({id_: i for i, id_ in enumerate(retrieved_ids)})
                    # results_df = results_df.sort_values('sort_order').drop('sort_order', axis=1)

                    result_count = len(results_df)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"I found {result_count} job(s) that seem relevant to '{query}':"
                    })
                    # Display results using the existing formatter
                    for index, job in results_df.iterrows():
                        format_job_display(job)
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Sorry, I couldn't find specific jobs matching '{query}'. You could try different keywords, rephrasing, or viewing all jobs."
                    })

        st.session_state.action = None # Reset action
        st.session_state.search_query = None # Clear search query
        st.rerun() # Rerun to show search results message

# --- Optional: Add Handling for General RAG Q&A ---
def handle_general_job_query(prompt):
    """Handles a general question using full RAG."""
    with st.spinner("Thinking..."):
        context = rag_service.get_context_text_for_rag(prompt, top_k=3)
        if not context:
            return "I couldn't find relevant information to answer that specific question right now."

        response = rag_service.generate_response_huggingface(prompt, context)
        return response