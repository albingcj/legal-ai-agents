# Legal AI Assistant

A multi-agent AI system for legal research and analysis using Streamlit.

## Quick Start

### 1. Activate Virtual Environment
```bash
# Windows
.\env\Scripts\activate

# Linux/Mac
source env/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run ui/streamlit_app.py
```

### 4. Access the Interface
Open your browser to `http://localhost:8501`

## Usage

1. Wait for the legal database to initialize (green status in sidebar)
2. Type your legal question in the input field
3. Click "Send" to get AI-powered legal analysis
4. View real-time processing status and detailed activity logs

## Features

- Multi-agent workflow (Research + Analysis)
- Real-time streaming responses
- Legal document search with RAG
- Conversation history
- Security and rate limiting

## Example Questions

- "What are the requirements for a valid contract?"
- "Can my employer terminate me without cause?"
- "How does copyright fair use work?"

---

**Disclaimer:** This tool provides legal information, not legal advice. Consult a qualified attorney for legal matters.
