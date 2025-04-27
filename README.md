# 🌟 Asha AI Career Assistant

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success)]()
[![Open Issues](https://img.shields.io/github/issues/jswalsakshi/asha-ai-chatbot)](https://github.com/jswalsakshi/asha-ai-chatbot/issues)

Asha is an **AI-powered career companion** built with Streamlit that provides personalized assistance for job seekers. The application offers a conversational interface with specialized modules for job searching, resume building, interview preparation, and professional mentorship guidance.

---

## 🚀 Features

### 🤖 Context-Aware AI Assistant
- Powered by **Gemini API** with context-specific prompting  
- **RAG (Retrieval-Augmented Generation)** for knowledge-based responses  
- Persistent conversation history with user authentication  

### 💼 Job Search
- Personalized job recommendations based on user profile and preferences  
- **Semantic search** using FAISS vector index  
- Interactive job browsing with direct application links  

### 📝 Resume Builder
- Guided resume creation through conversational interface  
- Professional PDF generation with customizable templates  
- One-click downloadable resumes  

### 🎤 Interview Preparation
- Contextual interview question practice  
- Industry-specific advice for technical and behavioral interviews  
- **STAR method** guidance for structured responses  

### 👩‍🏫 Mentorship Guidance
- Personalized mentor matching suggestions  
- Industry-specific mentorship advice  
- Networking strategies and professional development tips  

---

## 🛠️ Technical Implementation

- **Frontend**: Streamlit with custom UI components  
- **Backend**: Python services for RAG, LLM integration, and job recommendation  
- **Embedding**: BGE Small model for semantic search  
- **Document Generation**: FPDF for resume creation  
- **Authentication**: Simple hash-based user system with session management  

---

## ⚙️ Setup

1. **Clone the repository**

```bash
git clone https://github.com/jswalsakshi/asha-ai-chatbot.git
cd asha-ai-chatbot
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set environment variables in .env

ini
Copy
Edit
GEMINI_API_KEY=your_key
HUGGINGFACE_API_URL=your_url
HUGGINGFACE_API_TOKEN=your_token
Run the app

bash
Copy
Edit
streamlit run frontend/app.py
