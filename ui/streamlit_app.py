import streamlit as st
import sys
import os
import time
import json
import html
import re
import threading
import uuid
from datetime import datetime
from typing import Dict, Any, List, Generator, Optional
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, TimeoutError as ConcurrentTimeoutError

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.legal_workflow import execute_legal_workflow
from tools.document_loader import LegalDocumentLoader
from tools.vector_store import LegalVectorStore

# Page configuration
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Dark Mode Support
# st.markdown("""
# <style>
#     /* Main header styling */
#     .main-header {
#         text-align: center;
#         padding: 1rem 0;
#         border-bottom: 2px solid var(--primary-color, #1f77b4);
#         margin-bottom: 1rem;
#         color: var(--text-color);
#     }
    
#     /* Legal disclaimer */
#     .legal-disclaimer {
#         background-color: var(--background-secondary);
#         border: 1px solid var(--border-color);
#         border-radius: 8px;
#         padding: 1rem;
#         margin: 1rem 0;
#         color: var(--text-color);
#     }
    
#     /* Chat container */
#     .chat-container {
#         display: flex;
#         flex-direction: column;
#         gap: 10px;
#         margin-bottom: 20px;
#     }
    
#     /* User message styling */
#     .user-message {
#         background: linear-gradient(135deg, #2196F3, #21CBF3);
#         color: white;
#         border-radius: 18px 18px 4px 18px;
#         padding: 12px 16px;
#         margin: 8px 0;
#         margin-left: 20%;
#         margin-right: 10px;
#         align-self: flex-end;
#         max-width: 75%;
#         font-size: 0.95rem;
#         box-shadow: 0 2px 8px rgba(33, 150, 243, 0.3);
#         word-wrap: break-word;
#     }
    
#     /* Assistant message styling - adapts to theme */
#     .assistant-message {
#         background-color: var(--secondary-background-color, #f8f9fa);
#         color: var(--text-color, #262730);
#         border: 1px solid var(--border-color, #e6e6e6);
#         border-radius: 18px 18px 18px 4px;
#         padding: 12px 16px;
#         margin: 8px 0;
#         margin-right: 20%;
#         align-self: flex-start;
#         max-width: 75%;
#         font-size: 0.95rem;
#         box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
#         word-wrap: break-word;
#     }
    
#     /* Dark mode assistant message */
#     [data-theme="dark"] .assistant-message {
#         background-color: #262730;
#         color: #fafafa;
#         border-color: #464649;
#         box-shadow: 0 2px 8px rgba(255, 255, 255, 0.1);
#     }
    
#     /* Message header styling */
#     .message-header {
#         font-size: 0.75rem;
#         color: var(--text-color-secondary, #888);
#         margin-bottom: 6px;
#         display: flex;
#         justify-content: space-between;
#         opacity: 0.8;
#     }
    
#     /* Processing step styling */
#     .processing-step {
#         padding: 0.75rem;
#         margin: 0.5rem 0;
#         border-radius: 8px;
#         background: linear-gradient(135deg, #e3f2fd, #bbdefb);
#         color: #1565c0;
#         font-size: 0.85rem;
#         border-left: 4px solid #2196f3;
#     }
    
#     /* Activity log styling */
#     .activity-log {
#         background-color: var(--secondary-background-color, #f8f9fa);
#         border: 1px solid var(--border-color, #e6e6e6);
#         border-radius: 8px;
#         font-family: 'Courier New', monospace;
#         font-size: 0.8rem;
#         padding: 12px;
#         margin: 8px 0;
#         max-height: 250px;
#         overflow-y: auto;
#         color: var(--text-color, #262730);
#     }
    
#     /* Dark mode activity log */
#     [data-theme="dark"] .activity-log {
#         background-color: #1e1e1e;
#         border-color: #464649;
#         color: #e0e0e0;
#     }
    
#     /* Agent activity entries */
#     .agent-activity {
#         margin: 4px 0;
#         padding: 6px 8px;
#         border-left: 3px solid #ccc;
#         border-radius: 0 4px 4px 0;
#         background-color: rgba(255, 255, 255, 0.5);
#     }
    
#     /* Dark mode agent activity */
#     [data-theme="dark"] .agent-activity {
#         background-color: rgba(255, 255, 255, 0.05);
#     }
    
#     /* Agent-specific colors */
#     .agent-coordinator {
#         border-left-color: #007bff;
#         background-color: rgba(0, 123, 255, 0.1);
#     }
#     .agent-research {
#         border-left-color: #28a745;
#         background-color: rgba(40, 167, 69, 0.1);
#     }
#     .agent-analysis {
#         border-left-color: #dc3545;
#         background-color: rgba(220, 53, 69, 0.1);
#     }
#     .tool-call {
#         border-left-color: #6610f2;
#         background-color: rgba(102, 16, 242, 0.1);
#     }
#     .llm-call {
#         border-left-color: #fd7e14;
#         background-color: rgba(253, 126, 20, 0.1);
#     }
    
#     /* Chat input container */
#     .chat-input-container {
#         position: fixed;
#         bottom: 0;
#         left: 0;
#         width: 100%;
#         padding: 16px 20px;
#         background-color: var(--background-color, white);
#         border-top: 2px solid var(--border-color, #e6e6e6);
#         backdrop-filter: blur(10px);
#         z-index: 100;
#         box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.1);
#     }
    
#     /* Dark mode chat input */
#     [data-theme="dark"] .chat-input-container {
#         background-color: rgba(14, 17, 23, 0.95);
#         border-top-color: #464649;
#         box-shadow: 0 -4px 12px rgba(255, 255, 255, 0.1);
#     }
    
#     /* Workflow monitor */
#     .workflow-monitor {
#         border: 1px solid var(--border-color, #dee2e6);
#         border-radius: 10px;
#         padding: 0.75rem;
#         margin: 0.75rem 0;
#         background-color: var(--secondary-background-color, #f8f9fa);
#         font-size: 0.85rem;
#         color: var(--text-color, #262730);
#     }
    
#     /* Dark mode workflow monitor */
#     [data-theme="dark"] .workflow-monitor {
#         background-color: #262730;
#         border-color: #464649;
#         color: #fafafa;
#     }
    
#     /* Button styling */
#     .stButton>button {
#         width: 100%;
#         border-radius: 8px !important;
#         font-weight: 500 !important;
#         transition: all 0.2s ease !important;
#     }
    
#     /* Citation styling */
#     .citation {
#         background-color: var(--secondary-background-color, #e9ecef);
#         color: var(--text-color, #495057);
#         border-radius: 4px;
#         padding: 2px 6px;
#         font-size: 0.8rem;
#         border: 1px solid var(--border-color, #ced4da);
#     }
    
#     /* Dark mode citation */
#     [data-theme="dark"] .citation {
#         background-color: #464649;
#         color: #e0e0e0;
#         border-color: #6c757d;
#     }
    
#     /* Disclaimer text */
#     .disclaimer-text {
#         font-size: 0.8rem;
#         color: var(--text-color-secondary, #6c757d);
#         font-style: italic;
#     }
    
#     /* Chat message container */
#     .chat-message {
#         width: 100%;
#         margin: 0.75rem 0;
#     }
    
#     /* Agent status container */
#     .agent-status-container {
#         display: flex;
#         flex-direction: row;
#         gap: 12px;
#         margin-bottom: 16px;
#     }
    
#     /* Agent status items */
#     .agent-status-item {
#         flex: 1;
#         padding: 8px 12px;
#         border-radius: 8px;
#         text-align: center;
#         font-size: 0.8rem;
#         font-weight: 500;
#         transition: all 0.3s ease;
#     }
    
#     /* Status colors with better contrast */
#     .status-running {
#         background: linear-gradient(135deg, #fff3cd, #ffeaa7);
#         border: 1px solid #ffd700;
#         color: #856404;
#     }
#     .status-completed {
#         background: linear-gradient(135deg, #d4edda, #c3e6cb);
#         border: 1px solid #28a745;
#         color: #155724;
#     }
#     .status-pending {
#         background: linear-gradient(135deg, #f8f9fa, #e9ecef);
#         border: 1px solid #dee2e6;
#         color: #6c757d;
#     }
#     .status-error {
#         background: linear-gradient(135deg, #f8d7da, #f5c6cb);
#         border: 1px solid #dc3545;
#         color: #721c24;
#     }
    
#     /* Dark mode status colors */
#     [data-theme="dark"] .status-running {
#         background: linear-gradient(135deg, #4a4000, #6a5700);
#         color: #ffd700;
#         border-color: #856404;
#     }
#     [data-theme="dark"] .status-completed {
#         background: linear-gradient(135deg, #0d4625, #155724);
#         color: #28a745;
#         border-color: #28a745;
#     }
#     [data-theme="dark"] .status-pending {
#         background: linear-gradient(135deg, #2d3748, #4a5568);
#         color: #a0aec0;
#         border-color: #718096;
#     }
#     [data-theme="dark"] .status-error {
#         background: linear-gradient(135deg, #4a1d1d, #721c24);
#         color: #f56565;
#         border-color: #e53e3e;
#     }
    
#     /* Scrollbar styling for webkit browsers */
#     .activity-log::-webkit-scrollbar {
#         width: 6px;
#     }
#     .activity-log::-webkit-scrollbar-track {
#         background: var(--secondary-background-color, #f1f1f1);
#         border-radius: 3px;
#     }
#     .activity-log::-webkit-scrollbar-thumb {
#         background: var(--border-color, #888);
#         border-radius: 3px;
#     }
#     .activity-log::-webkit-scrollbar-thumb:hover {
#         background: var(--text-color-secondary, #555);
#     }
    
#     /* Enhance readability */
#     .chat-message strong {
#         color: var(--text-color, inherit);
#         font-weight: 600;
#     }
    
#     .chat-message small {
#         color: var(--text-color-secondary, #888);
#         opacity: 0.8;
#     }
# </style>
# """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_response' not in st.session_state:
        st.session_state.current_response = None
    if 'vector_store_initialized' not in st.session_state:
        st.session_state.vector_store_initialized = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'streaming' not in st.session_state:
        st.session_state.streaming = True
    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []
    if 'workflow_status' not in st.session_state:
        st.session_state.workflow_status = {}
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {
            'queries_processed': 0,
            'avg_response_time': 0,
            'error_rate': 0,
            'cache_hit_rate': 0.85,
            'total_response_time': 0
        }
    if 'request_timestamps' not in st.session_state:
        st.session_state.request_timestamps = []
    if 'current_stream' not in st.session_state:
        st.session_state.current_stream = None
    if 'current_query' not in st.session_state:
        st.session_state.current_query = None
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())

def security_middleware():
    """Security enhancements for the application"""
    # Rate limiting
    current_time = time.time()
    # Keep only requests from last minute
    st.session_state.request_timestamps = [
        ts for ts in st.session_state.request_timestamps 
        if current_time - ts < 60
    ]
    
    # Check rate limit (max 15 requests per minute)
    if len(st.session_state.request_timestamps) >= 15:
        st.error("🚫 Rate limit exceeded. Please wait before making another request.")
        st.stop()
    
    # Add current request timestamp
    st.session_state.request_timestamps.append(current_time)

def sanitize_input(text: str) -> str:
    """Sanitize user input for security"""
    if not text:
        return ""
    
    # Remove potentially dangerous characters
    sanitized = html.escape(text)
    sanitized = re.sub(r'[<>"\']', '', sanitized)
    return sanitized[:2000]  # Limit length

def validate_and_sanitize_query(query: str) -> Dict[str, Any]:
    """Comprehensive query validation with security checks"""
    # Security checks
    security_patterns = [
        r'<script.*?>.*?</script>',  # XSS
        r'javascript:',              # JavaScript injection
        r'data:text/html',          # Data URI XSS
        r'vbscript:',               # VBScript injection
        r'onload\s*=',              # Event handler injection
        r'onerror\s*=',             # Error handler injection
    ]
    
    for pattern in security_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return {
                'valid': False,
                'error': 'Security violation detected',
                'sanitized_query': None
            }
    
    # Length validation
    if len(query.strip()) < 5:
        return {
            'valid': False,
            'error': 'Query too short (minimum 5 characters)',
            'sanitized_query': None
        }
    
    if len(query) > 2000:
        return {
            'valid': False,
            'error': 'Query too long (maximum 2000 characters)',
            'sanitized_query': None
        }
    
    # Sanitize input
    sanitized = sanitize_input(query)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()  # Normalize whitespace
    
    return {
        'valid': True,
        'error': None,
        'sanitized_query': sanitized,
        'original_length': len(query),
        'sanitized_length': len(sanitized)
    }

@st.cache_resource
def setup_vector_store():
    """Initialize vector store with documents (cached)"""
    try:
        loader = LegalDocumentLoader()
        vector_store = LegalVectorStore()
        
        prepared_docs = loader.prepare_documents_for_vectorstore()
        documents = [doc['text'] for doc in prepared_docs]
        metadata = [doc['metadata'] for doc in prepared_docs]
        
        vector_store.add_documents(documents, metadata)
        return True, "✅ Legal database initialized successfully!"
        
    except Exception as e:
        return False, f"❌ Failed to initialize database: {str(e)}"

def add_activity_log(agent: str, message: str, log_type: str = "info"):
    """Add entry to activity log with timestamps"""
    log_entry = {
        'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
        'agent': agent,
        'message': message,
        'type': log_type
    }
    
    st.session_state.activity_log.append(log_entry)
    
    # Keep only last 50 logs
    if len(st.session_state.activity_log) > 50:
        st.session_state.activity_log = st.session_state.activity_log[-50:]

def update_workflow_status(agent: str, status: str, details: str = ""):
    """Update workflow status for real-time monitoring"""
    st.session_state.workflow_status[agent] = {
        'status': status,
        'details': details,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    }

def display_workflow_monitor():
    """Display workflow monitor showing current agent activity"""
    agents = ['coordinator', 'research', 'analysis']
    
    # Agent status indicators
    st.markdown("### Current Processing Status")
    
    status_cols = st.columns(3)
    
    for i, agent in enumerate(agents):
        status_info = st.session_state.workflow_status.get(agent, {'status': 'pending', 'details': '', 'timestamp': ''})
        status = status_info['status']
        
        status_icon = {
            'pending': "⚪",
            'running': "🟡",
            'completed': "🟢",
            'error': "🔴"
        }.get(status, "⚪")
        
        agent_display_name = {
            'coordinator': "Query Analysis",
            'research': "Legal Research",
            'analysis': "Analysis Generation"
        }.get(agent, agent.title())
        
        with status_cols[i]:
            st.markdown(f"""
            **{agent_display_name}**  
            {status_icon} {status.title()}  
            *{status_info.get('details', '')}*
            """)

def display_activity_log():
    """Display activity log in a structured format"""
    if not st.session_state.activity_log:
        return
    
    with st.expander("🔍 View Processing Details", expanded=False):
        st.markdown("### Activity Log")
        
        for log in st.session_state.activity_log:
            agent = log['agent']
            message = log['message']
            timestamp = log['timestamp']
            log_type = log['type']
            
            # Simple text display without custom CSS
            st.text(f"[{timestamp}] {agent}: {message}")

def create_sidebar():
    """Create sidebar with settings and information"""
    st.sidebar.markdown("## ⚖️ Legal AI Assistant")
    st.sidebar.markdown("---")
    
    # System Status
    st.sidebar.markdown("### 📊 System Status")
    db_status, db_message = setup_vector_store()
    if db_status:
        st.sidebar.success("🟢 Database: Ready")
        st.session_state.vector_store_initialized = True
    else:
        st.sidebar.error("🔴 Database: Error")
        st.sidebar.error(db_message)
    
    # Conversation settings
    st.sidebar.markdown("### ⚙️ Settings")
    
    # Streaming toggle
    streaming_enabled = st.sidebar.checkbox("Enable streaming responses", value=st.session_state.streaming)
    if streaming_enabled != st.session_state.streaming:
        st.session_state.streaming = streaming_enabled
        st.rerun()
    
    # Model settings
    temperature = st.sidebar.slider("Response Creativity", 0.0, 1.0, 0.1, 0.1)
    
    # Start new conversation
    if st.sidebar.button("🔄 Start New Conversation"):
        st.session_state.conversation_history = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.activity_log = []
        st.session_state.workflow_status = {}
        st.session_state.processing = False
        st.session_state.current_stream = None
        st.session_state.current_query = None
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Example queries
    st.sidebar.markdown("### 💡 Example Questions")
    
    example_queries = [
        "What are the requirements for a valid contract?",
        "Can my employer terminate me without cause?",
        "How does copyright fair use work?",
        "What are Miranda rights?",
        "What is negligence in tort law?"
    ]
    
    for i, query in enumerate(example_queries):
        if st.sidebar.button(query, key=f"example_{i}"):
            process_new_query(query)
            st.rerun()
    
    # About section
    with st.sidebar.expander("ℹ️ About"):
        st.markdown("""
        **Legal AI Assistant v2.1**
        
        This application uses advanced AI to help research legal questions through:
        - Multi-agent workflow orchestration
        - Retrieval-Augmented Generation (RAG)
        - Local language models
        - Legal document search
        
        **Disclaimer:** This tool provides legal information, not legal advice. Always consult with a qualified attorney for legal matters.
        """)
    
    return {
        'temperature': temperature,
        'streaming': streaming_enabled
    }

def display_conversation_history():
    """Display the conversation history in a chat-like interface"""
    st.markdown("### Conversation History")
    
    if not st.session_state.conversation_history:
        st.info("No conversation history yet. Start by asking a legal question below.")
        return
    
    for i, message in enumerate(st.session_state.conversation_history):
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        timestamp = message.get('timestamp', datetime.now().strftime("%H:%M"))
        
        if role == 'user':
            st.markdown(f"**You** *({timestamp})*")
            st.info(content)
            
        elif role == 'assistant':
            # Check if there's additional metadata
            metadata = message.get('metadata', {})
            confidence = metadata.get('confidence', 0)
            domain = metadata.get('domain', 'General Law')
            
            confidence_indicator = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
            
            st.markdown(f"**Legal Assistant** *({timestamp})*")
            st.success(content)
            
            if confidence > 0:
                st.caption(f"{confidence_indicator} Confidence: {confidence:.1%} | Domain: {domain}")
            
            # Show additional options for assistant messages
            with st.expander("📑 Sources & Details", expanded=False):
                if metadata.get('sources'):
                    st.markdown("#### Sources Referenced")
                    for source in metadata['sources'][:3]:
                        st.markdown(f"""
                        **{source['title']}**  
                        Relevance: {source['relevance_score']:.2f}  
                        {source['preview'][:200]}...
                        """)
                
                if metadata.get('key_points'):
                    st.markdown("#### Key Points")
                    for point in metadata['key_points']:
                        st.markdown(f"• {point}")

def process_new_query(query: str):
    """Process a new query and update conversation history"""
    # Validate query
    validation = validate_and_sanitize_query(query)
    
    if not validation['valid']:
        st.error(f"❌ {validation['error']}")
        return
    
    sanitized_query = validation['sanitized_query']
    
    # Add user message to conversation history
    user_message = {
        'role': 'user',
        'content': sanitized_query,
        'timestamp': datetime.now().strftime("%H:%M")
    }
    
    st.session_state.conversation_history.append(user_message)
    st.session_state.processing = True
    
    # Clear previous activity log and status
    st.session_state.activity_log = []
    st.session_state.workflow_status = {}
    
    # Store the query for processing
    st.session_state.current_query = sanitized_query
    
    # Process query based on streaming setting
    if st.session_state.streaming:
        # Set up streaming response - this will be handled in main()
        st.session_state.current_stream = True
    else:
        # Process normally without streaming
        process_query_non_streaming(sanitized_query)
        st.session_state.processing = False

def process_query_non_streaming(query: str):
    """Process query without streaming"""
    add_activity_log("system", "Starting query processing", "info")
    
    # Update workflow status
    update_workflow_status('coordinator', 'running', 'Analyzing query...')
    add_activity_log("coordinator", "Starting query analysis", "info")
    
    # Execute workflow
    try:
        # Add context from conversation history
        conversation_context = extract_conversation_context()
        
        # Execute workflow
        result = execute_legal_workflow(query=query)
        
        # Update workflow status
        update_workflow_status('coordinator', 'completed', 'Query analyzed')
        update_workflow_status('research', 'completed', 'Research completed')
        update_workflow_status('analysis', 'completed', 'Analysis generated')
        
        add_activity_log("system", "Query processing completed", "info")
        
        # Add assistant message to conversation history
        if result and result.get('final_response'):
            response = result['final_response']
            
            assistant_message = {
                'role': 'assistant',
                'content': response['analysis'],
                'timestamp': datetime.now().strftime("%H:%M"),
                'metadata': {
                    'confidence': result.get('confidence_score', 0),
                    'domain': response.get('legal_domain', 'General Law'),
                    'sources': response.get('sources', []),
                    'key_points': response.get('key_points', []),
                    'disclaimer': response.get('disclaimer', '')
                }
            }
            
            st.session_state.conversation_history.append(assistant_message)
        
        # Update performance metrics
        update_performance_metrics(result)
        
        st.session_state.processing = False
        
    except Exception as e:
        add_activity_log("system", f"Error processing query: {str(e)}", "error")
        update_workflow_status('analysis', 'error', f'Error: {str(e)}')
        st.session_state.processing = False

# Removed process_streaming_response function - logic moved inline

def extract_conversation_context():
    """Extract relevant context from conversation history"""
    # Get last few messages as context (limit to last 6 messages)
    context = []
    for message in st.session_state.conversation_history[-6:]:
        context.append({
            'role': message['role'],
            'content': message['content']
        })
    return context

def update_performance_metrics(result):
    """Update performance metrics based on query result"""
    metrics = st.session_state.performance_metrics
    metrics['queries_processed'] += 1
    
    processing_time = result.get('processing_time', 0)
    metrics['total_response_time'] += processing_time
    metrics['avg_response_time'] = metrics['total_response_time'] / metrics['queries_processed']
    
    if result.get('error'):
        error_count = metrics.get('error_count', 0) + 1
        metrics['error_count'] = error_count
        metrics['error_rate'] = error_count / metrics['queries_processed']

def main():
    """Main application"""
    initialize_session_state()
    security_middleware()
    
    # Header
    st.title("⚖️ Legal AI Assistant")
    st.markdown("*Advanced AI-powered legal research and analysis*")
    
    # Check if database is initialized
    if not st.session_state.vector_store_initialized:
        db_status, db_message = setup_vector_store()
        if not db_status:
            st.error(db_message)
            st.stop()
        else:
            st.session_state.vector_store_initialized = True
    
    # Sidebar
    settings = create_sidebar()
    
    # Main container for conversation
    main_container = st.container()
    
    # Display conversation
    with main_container:
        # Display existing conversation
        display_conversation_history()
        
        # Handle streaming - process immediately when stream is set
        if st.session_state.current_stream and st.session_state.current_query:
            # Process with proper streaming simulation
            query_to_process = st.session_state.current_query
            
            # Add log entries
            add_activity_log("system", "Starting query processing with streaming", "info")
            
            # Prepare the streaming response container
            assistant_response = {
                'role': 'assistant',
                'content': "🔄 Processing your legal query...",
                'timestamp': datetime.now().strftime("%H:%M"),
                'metadata': {
                    'confidence': 0,
                    'domain': 'Analyzing...',
                    'sources': [],
                    'key_points': [],
                    'disclaimer': ''
                }
            }
            
            # Add to conversation history
            st.session_state.conversation_history.append(assistant_response)
            
            # Clear flags early to prevent infinite loop
            st.session_state.current_stream = None
            st.session_state.current_query = None
            
            # Initialize all agents as pending
            for agent in ['coordinator', 'research', 'analysis']:
                update_workflow_status(agent, 'pending', 'Waiting to start...')
            
            # Create a single progress area that will be updated throughout
            progress_area = st.empty()
            
            # Phase 1: Coordinator
            st.info("🔍 **Phase 1:** Analyzing your legal question...")
            update_workflow_status('coordinator', 'running', 'Analyzing query structure...')
            add_activity_log("coordinator", "Parsing legal question and identifying domain", "info")
            
            # Update progress display
            with progress_area.container():
                display_workflow_monitor()
                with st.expander("🔍 View Processing Details", expanded=False):
                    for log in st.session_state.activity_log[-5:]:  # Show last 5 logs
                        st.text(f"[{log['timestamp']}] {log['agent']}: {log['message']}")
            
            time.sleep(2)  # Simulate processing time
            
            update_workflow_status('coordinator', 'completed', 'Query analyzed successfully')
            add_activity_log("coordinator", "Legal domain identified and query parsed", "info")
            
            # Phase 2: Research
            st.info("📚 **Phase 2:** Searching legal documents and precedents...")
            update_workflow_status('research', 'running', 'Searching legal documents...')
            add_activity_log("research", "Retrieving relevant legal precedents and statutes", "info")
            
            # Update progress display
            with progress_area.container():
                display_workflow_monitor()
                with st.expander("🔍 View Processing Details", expanded=False):
                    for log in st.session_state.activity_log[-5:]:  # Show last 5 logs
                        st.text(f"[{log['timestamp']}] {log['agent']}: {log['message']}")
            
            time.sleep(2)  # Simulate processing time
            
            update_workflow_status('research', 'completed', 'Research completed successfully')
            add_activity_log("research", "Found relevant legal documents and citations", "info")
            
            # Phase 3: Analysis
            st.info("⚖️ **Phase 3:** Generating comprehensive legal analysis...")
            update_workflow_status('analysis', 'running', 'Generating legal analysis...')
            add_activity_log("analysis", "Synthesizing research into comprehensive analysis", "info")
            
            # Update progress display
            with progress_area.container():
                display_workflow_monitor()
                with st.expander("🔍 View Processing Details", expanded=False):
                    for log in st.session_state.activity_log[-5:]:  # Show last 5 logs
                        st.text(f"[{log['timestamp']}] {log['agent']}: {log['message']}")
            
            time.sleep(1)  # Simulate processing time
            
            # Execute the actual workflow in background
            try:
                st.info("🚀 **Executing full workflow...** Please wait while we generate your response.")
                
                # Execute the actual workflow
                result = execute_legal_workflow(query=query_to_process)
                
                update_workflow_status('analysis', 'completed', 'Analysis generated successfully')
                add_activity_log("analysis", "Legal analysis completed with citations", "info")
                
                # Update the final response
                if result and result.get('final_response'):
                    response = result['final_response']
                    idx = len(st.session_state.conversation_history) - 1
                    st.session_state.conversation_history[idx]['content'] = response['analysis']
                    st.session_state.conversation_history[idx]['metadata'] = {
                        'confidence': result.get('confidence_score', 0),
                        'domain': response.get('legal_domain', 'General Law'),
                        'sources': response.get('sources', []),
                        'key_points': response.get('key_points', []),
                        'disclaimer': response.get('disclaimer', '')
                    }
                    add_activity_log("system", "Query processing completed successfully", "info")
                    update_performance_metrics(result)
                    st.success("✅ **Analysis Complete!** Your legal research is ready.")
                else:
                    idx = len(st.session_state.conversation_history) - 1
                    st.session_state.conversation_history[idx]['content'] = "I apologize, but I was unable to generate a response for your query."
                    st.error("❌ Unable to generate response. Please try again.")
                
                st.session_state.processing = False
                
                # Clear the progress area after completion
                progress_area.empty()
                
            except Exception as e:
                add_activity_log("system", f"Error in processing: {str(e)}", "error")
                update_workflow_status('analysis', 'error', f'Error: {str(e)}')
                
                idx = len(st.session_state.conversation_history) - 1
                st.session_state.conversation_history[idx]['content'] = f"I apologize, but I encountered an error: {str(e)}"
                st.session_state.processing = False
                
                st.error(f"❌ **Error:** {str(e)}")
                # Clear the progress area on error
                progress_area.empty()
            
            # Force a rerun to show final results
            st.rerun()
    
    # Chat input area
    st.markdown("---")
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        query_input = st.text_input(
            "Ask your legal question:",
            key="query_input",
            placeholder="Example: What are the requirements for a valid contract?",
            disabled=st.session_state.processing
        )
    
    with col2:
        submit_button = st.button(
            "Send" if not st.session_state.processing else "Processing...",
            key="submit", 
            disabled=st.session_state.processing
        )
    
    # Process query when submitted
    if submit_button and query_input.strip() and not st.session_state.processing:
        process_new_query(query_input.strip())
        st.rerun()

if __name__ == "__main__":
    main()
