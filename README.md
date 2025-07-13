# ⚖️ Legal AI Agent

A powerful AI-powered legal research and analysis assistant that helps users understand legal concepts across multiple domains including Contract Law, Employment Law, Intellectual Property, Criminal Law, and Tort Law.

## �️ Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Frontend** | [Streamlit](https://streamlit.io/) | Web interface and user interaction |
| **AI Orchestration** | [LangGraph](https://langchain-ai.github.io/langgraph/) | Workflow management and agent coordination |
| **LLM Framework** | [LangChain](https://langchain.com/) | Large language model integration |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | Document storage and similarity search |
| **Embeddings** | [SentenceTransformers](https://www.sbert.net/) | Text embedding generation |
| **Local AI** | [LM Studio](https://lmstudio.ai/) | Local LLM hosting and inference |
| **Language** | Python 3.11+ | Core development language |

### 🔧 Key Features:
- **Multi-Agent Architecture**: Coordinator, Research, and Analysis agents working together
- **Local AI Processing**: No data sent to external APIs (privacy-focused)
- **Vector Search**: Intelligent document retrieval using semantic similarity
- **Auto-Initialization**: Database setup happens automatically on first run
- **Thread-Safe**: Concurrent request handling with retry mechanisms

## �🚀 Quick Start

### Prerequisites
- Python 3.11+
- LM Studio (for local AI model)
- Git

### 1. Clone and Setup
```bash
git clone https://github.com/albingcj/legal-ai-agents.git
cd legal-ai-agents
```

### 2. Create Virtual Environment
```bash
python -m venv env
```

### 3. Activate Environment
**Windows (PowerShell):**
```powershell
.\env\Scripts\activate
```

**Windows (Command Prompt):**
```cmd
env\Scripts\activate.bat
```

**Mac/Linux:**
```bash
source env/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Environment Configuration
Create a `.env` file in the project root:
```env
# LM Studio Configuration (Optional - uses defaults if not set)
LM_STUDIO_BASE_URL=http://localhost:1234
LM_STUDIO_API_KEY=not-needed
MODEL_NAME=microsoft/Phi-3-mini-4k-instruct-gguf

# Database Configuration (Optional - uses defaults if not set)
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_RETRIEVED_DOCS=8
TEMPERATURE=0.1
DEBUG=False
```

### 6. Setup LM Studio (Local AI Model)
1. Download and install [LM Studio](https://lmstudio.ai/) or you can use local Llama or any other LLMs
2. Download a compatible model (recommended: `microsoft/Phi-3-mini-4k-instruct-gguf`)
3. Start the local server in LM Studio on port 1234

### 7. Run the Application
```bash
streamlit run app.py
```

The app will automatically:
- ✅ Initialize the legal document database on first run
- 📊 Load sample legal documents into the vector database
- 🚀 Start the web interface at `http://localhost:8501`

> **Note**: The first startup may take a few extra seconds to initialize the database. Subsequent runs will be faster as the database is already set up.

## 📚 Adding More Legal Documents

### Method 1: Using the Document Loader (Recommended)
Add documents programmatically by modifying the sample documents in `tools/document_loader.py`:

```python
# Add your documents to the sample_docs list in load_sample_documents()
{
    "title": "Your Legal Topic",
    "content": """
    Your legal content here...
    """,
    "source": "Source Reference",
    "type": "legal_principle",  # or "case_law", "statute", etc.
    "domain": "contract_law"  # or other supported domain
}
```

Then restart the application or manually reinitialize:
```bash
# Option 1: Restart the app (it will detect and reload documents)
streamlit run app.py

# Option 2: Manual reinitialize (if needed)
python -c "
from tools.document_loader import LegalDocumentLoader
from tools.vector_store import LegalVectorStore

loader = LegalDocumentLoader()
vector_store = LegalVectorStore()
vector_store.reset_collection()  # Clear existing data

prepared_docs = loader.prepare_documents_for_vectorstore()
documents = [doc['text'] for doc in prepared_docs]
metadata = [doc['metadata'] for doc in prepared_docs]

vector_store.add_documents(documents, metadata)
print('Database updated successfully!')
"
```

### Method 2: Direct Database Addition
```python
from tools.vector_store import LegalVectorStore

vector_store = LegalVectorStore()

# Add single document
vector_store.add_documents(
    documents=["Your legal document content"],
    metadata=[{
        "title": "Document Title",
        "source": "Source",
        "domain": "contract_law",
        "type": "legal_principle"
    }]
)
```

### Method 3: File-based Loading
Create text files in the `data/legal_documents/` directory and use:
```python
from tools.document_loader import LegalDocumentLoader

loader = LegalDocumentLoader()
docs = loader.load_documents_from_directory("data/legal_documents/")
# Process and add to vector store...
```

## 🏗️ Project Structure

```
legal-ai-agents/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment configuration
├── README.md             # This file
│
├── agents/               # AI agent modules
│   ├── coordinator.py    # Query coordination
│   ├── research_agent.py # Legal research
│   └── analysis_agent.py # Legal analysis
│
├── workflows/            # Workflow orchestration
│   └── legal_workflow.py # Main workflow logic
│
├── tools/                # Utility tools
│   ├── document_loader.py # Document loading
│   ├── vector_store.py   # Vector database
│   └── llm_client.py     # LLM client
│
├── config/               # Configuration
│   └── settings.py       # App settings
│
├── chroma_db/            # Vector database (auto-created)
├── data/                 # Document storage (auto-created)
└── tests/                # Test files
```

## 🎯 Supported Legal Domains

- **Contract Law**: Formation, breach, remedies
- **Employment Law**: Workplace rights, termination, discrimination
- **Intellectual Property**: Copyright, patents, trademarks
- **Criminal Law**: Procedures, rights, defenses
- **Tort Law**: Negligence, liability, damages

## 🔧 Configuration Options

Edit `.env` file or `config/settings.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `LM_STUDIO_BASE_URL` | `http://localhost:1234` | LM Studio server URL |
| `CHROMA_DB_PATH` | `./chroma_db` | Vector database location |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `MAX_RETRIEVED_DOCS` | `8` | Max documents per query |
| `TEMPERATURE` | `0.1` | AI response creativity |

## 🧪 Testing

Run tests to verify functionality:
```bash
# Test individual components
python -m pytest tests/

# Test complete system
python tests/test_complete_system.py

# Test workflow manually
python workflows/legal_workflow.py
```

## 🔍 Usage Examples

### Example Queries:
- "What are the requirements for a valid contract?"
- "Can my employer terminate me without cause?"
- "How does copyright fair use work?"
- "What are Miranda rights in criminal law?"
- "When is someone liable for negligence?"

## 🛠️ Troubleshooting

### Common Issues:

**1. ModuleNotFoundError: No module named 'agents'**
- Make sure you're running from the project root directory
- Use `streamlit run app.py` instead of running workflow files directly

**2. LM Studio Connection Error**
- Ensure LM Studio is running on port 1234
- Check that a model is loaded in LM Studio
- Verify the `LM_STUDIO_BASE_URL` in your `.env` file

**3. Database Issues**
- The app automatically initializes the database on first run
- If you see database errors, delete the `chroma_db` folder and restart the app
- Check that the project directory has write permissions

**4. Dependencies Issues**
- Ensure you're in the virtual environment: `.\env\Scripts\activate`
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

### Reset Everything:
```bash
# Clear database (Windows PowerShell)
Remove-Item -Recurse -Force chroma_db

# Or for Unix/Mac/Linux
rm -rf chroma_db/

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Restart the app (database will be automatically reinitialized)
streamlit run app.py
```

---

**Built with**: Python, Streamlit, LangGraph, ChromaDB, SentenceTransformers, LM Studio