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
from tools.llm_client import UniversalLLMClient

# Page configuration
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_response' not in st.session_state:
        st.session_state.current_response = None
    if 'vector_store_initialized' not in st.session_state:
        st.session_state.vector_store_initialized = False
    if 'llm_initialized' not in st.session_state:
        st.session_state.llm_initialized = False
    if 'llm_status' not in st.session_state:
        st.session_state.llm_status = None
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

@st.cache_resource
def check_llm_health():
    """Check if LLM API is healthy and responding (cached)"""
    try:
        client = UniversalLLMClient()
        health_result = client.health_check()
        
        if health_result['status'] == 'healthy':
            return True, f"✅ LLM API is healthy ({health_result['provider']}: {health_result['model']})", health_result
        else:
            return False, f"❌ LLM API issue: {health_result['message']}", health_result
            
    except Exception as e:
        return False, f"❌ Failed to check LLM health: {str(e)}", None

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
    
    # Database status
    db_status, db_message = setup_vector_store()
    if db_status:
        st.sidebar.success("🟢 Database: Ready")
        st.session_state.vector_store_initialized = True
    else:
        st.sidebar.error("🔴 Database: Error")
        st.sidebar.error(db_message)
    
    # LLM API status
    llm_status, llm_message, llm_details = check_llm_health()
    if llm_status:
        st.sidebar.success("🟢 LLM API: Ready")
        st.session_state.llm_initialized = True
        st.session_state.llm_status = llm_details
        
        # Show LLM details in expander
        with st.sidebar.expander("🤖 LLM Details", expanded=False):
            if llm_details:
                st.write(f"**Provider:** {llm_details.get('provider', 'Unknown')}")
                st.write(f"**Model:** {llm_details.get('model', 'Unknown')}")
                st.write(f"**Full Model:** {llm_details.get('full_model', 'Unknown')}")
                if llm_details.get('api_base'):
                    st.write(f"**API Base:** {llm_details['api_base']}")
                if llm_details.get('response_time'):
                    st.write(f"**Response Time:** {llm_details['response_time']}s")
                if llm_details.get('test_response'):
                    st.write(f"**Test Response:** {llm_details['test_response']}")
                if llm_details.get('usage'):
                    st.write(f"**Usage:** {llm_details['usage']}")
    else:
        st.sidebar.error("🔴 LLM API: Error")
        st.sidebar.error(llm_message)
        st.session_state.llm_initialized = False
        st.session_state.llm_status = llm_details
        
        # Show troubleshooting info
        with st.sidebar.expander("🔧 Troubleshooting", expanded=True):
            if llm_details and llm_details.get('provider'):
                provider = llm_details['provider']
                if provider == 'openai' and 'localhost' in str(llm_details.get('api_base', '')):
                    st.markdown("""
                    **LM Studio / Local Model:**
                    1. Ensure LM Studio is running
                    2. Check if a model is loaded
                    3. Verify server is started (port 1234)
                    4. Check firewall settings
                    """)
                elif provider == 'openai':
                    st.markdown("""
                    **OpenAI:**
                    1. Check your API key
                    2. Verify internet connection
                    3. Check rate limits
                    4. Ensure model access
                    """)
                elif provider == 'anthropic':
                    st.markdown("""
                    **Anthropic:**
                    1. Check your API key
                    2. Verify internet connection
                    3. Check rate limits
                    4. Ensure model access
                    """)
                else:
                    st.markdown(f"""
                    **{provider.title()}:**
                    1. Check your API key
                    2. Verify internet connection
                    3. Check rate limits
                    4. Ensure model access
                    """)
            else:
                st.markdown("""
                **General troubleshooting:**
                1. Check API configuration
                2. Verify internet connection
                3. Check rate limits
                4. Ensure model access
                """)
            
            if llm_details:
                st.write(f"**Provider:** {llm_details.get('provider', 'Unknown')}")
                st.write(f"**Model:** {llm_details.get('full_model', 'Unknown')}")
                if llm_details.get('api_base'):
                    st.write(f"**API Base:** {llm_details['api_base']}")
                if llm_details.get('error_details'):
                    st.write(f"**Error:** {llm_details['error_details']}")
    
    # Overall system status
    system_ready = db_status and llm_status
    if system_ready:
        st.sidebar.success("🚀 **System Ready**")
    else:
        st.sidebar.warning("⚠️ **System Not Ready**")
        st.sidebar.info("Please fix the issues above before using the assistant.")
    
    # Add refresh button for system status
    if st.sidebar.button("🔄 Refresh System Status"):
        # Clear the cached functions to force recheck
        setup_vector_store.clear()
        check_llm_health.clear()
        st.session_state.vector_store_initialized = False
        st.session_state.llm_initialized = False
        st.session_state.llm_status = None
        st.rerun()
    
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
    
    # Check if systems are initialized
    if not st.session_state.vector_store_initialized:
        db_status, db_message = setup_vector_store()
        if not db_status:
            st.error("**Database Error:** " + db_message)
            st.info("Please check the database configuration and try refreshing the page.")
            st.stop()
        else:
            st.session_state.vector_store_initialized = True
    
    if not st.session_state.llm_initialized:
        llm_status, llm_message, llm_details = check_llm_health()
        if not llm_status:
            st.error("**LLM API Error:** " + llm_message)
            
            # Show detailed error information
            if llm_details:
                with st.expander("🔧 Detailed Error Information", expanded=True):
                    st.write(f"**Provider:** {llm_details.get('provider', 'Unknown')}")
                    st.write(f"**Model:** {llm_details.get('full_model', 'Unknown')}")
                    if llm_details.get('api_base'):
                        st.write(f"**API Base:** {llm_details['api_base']}")
                    if llm_details.get('error_details'):
                        st.write(f"**Technical Error:** {llm_details['error_details']}")
                    
                    provider = llm_details.get('provider', '').lower()
                    
                    if provider == 'openai' and 'localhost' in str(llm_details.get('api_base', '')):
                        st.markdown("""
                        ### LM Studio / Local Model Setup:
                        1. **Download and Install LM Studio** from https://lmstudio.ai/
                        2. **Download a Model** - Search and download a chat model (e.g., Llama, Mistral)
                        3. **Load the Model** - Go to "Chat" tab and select your model
                        4. **Start Local Server** - Go to "Local Server" tab and click "Start Server"
                        5. **Configure Model** - Ensure the model is set to serve on port 1234
                        6. **Test Connection** - Try the server endpoint in your browser: http://localhost:1234/v1/models
                        """)
                    elif provider == 'openai':
                        st.markdown("""
                        ### OpenAI API Setup:
                        1. **Get API Key** - Visit https://platform.openai.com/api-keys
                        2. **Set Environment Variable** - Add `LLM_API_KEY=your-key-here` to your .env file
                        3. **Set Model** - Add `LLM_MODEL=openai/gpt-3.5-turbo` to your .env file
                        4. **Check Billing** - Ensure you have credits in your OpenAI account
                        """)
                    elif provider == 'anthropic':
                        st.markdown("""
                        ### Anthropic API Setup:
                        1. **Get API Key** - Visit https://console.anthropic.com/
                        2. **Set Environment Variable** - Add `LLM_API_KEY=your-key-here` to your .env file
                        3. **Set Model** - Add `LLM_MODEL=anthropic/claude-3-sonnet-20240229` to your .env file
                        4. **Check Credits** - Ensure you have credits in your Anthropic account
                        """)
                    else:
                        st.markdown(f"""
                        ### {provider.title()} API Setup:
                        1. **Get API Key** - Visit your provider's website
                        2. **Set Environment Variable** - Add `LLM_API_KEY=your-key-here` to your .env file
                        3. **Set Model** - Add `LLM_MODEL={provider}/model-name` to your .env file
                        4. **Check Documentation** - Visit LiteLLM docs for provider-specific setup
                        """)
                    
                    st.markdown("""
                    ### Environment Configuration (.env file):
                    ```
                    # Choose ONE of the following configurations:
                    
                    # For Google Gemini (Latest AI - Recommended):
                    LLM_MODEL=gemini/gemini-2.0-flash
                    LLM_API_KEY=your-google-api-key
                    
                    # For LM Studio (Local/Privacy):
                    LLM_MODEL=openai/local-model
                    LM_STUDIO_BASE_URL=http://localhost:1234
                    LM_STUDIO_API_KEY=not-needed
                    
                    # For OpenAI:
                    LLM_MODEL=openai/gpt-3.5-turbo
                    LLM_API_KEY=your-openai-api-key
                    
                    # For Anthropic:
                    LLM_MODEL=anthropic/claude-3-sonnet-20240229
                    LLM_API_KEY=your-anthropic-api-key
                    ```
                    """)
            
            st.info("The Legal AI Assistant supports multiple LLM providers via LiteLLM. Please configure your preferred provider and refresh the page.")
            st.stop()
        else:
            st.session_state.llm_initialized = True
            st.session_state.llm_status = llm_details
    
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
            # Process with real streaming from backend
            query_to_process = st.session_state.current_query
            
            # Add log entries
            add_activity_log("system", "Starting query processing with streaming", "info")
            
            # Prepare the streaming response container
            assistant_response = {
                'role': 'assistant',
                'content': "",
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
            
            # Create containers for streaming
            progress_area = st.empty()
            response_area = st.empty()
            
            try:
                # Import the streaming workflow
                from workflows.legal_workflow import execute_legal_workflow_stream
                
                # Start streaming workflow
                for stream_chunk in execute_legal_workflow_stream(query_to_process):
                    
                    if stream_chunk['type'] == 'status_update':
                        # Update agent status
                        agent = stream_chunk['agent']
                        status = stream_chunk['status']
                        message = stream_chunk['message']
                        
                        update_workflow_status(agent, status, message)
                        add_activity_log(agent, message, "info")
                        
                        # Update progress display
                        with progress_area.container():
                            display_workflow_monitor()
                            with st.expander("🔍 View Processing Details", expanded=False):
                                for log in st.session_state.activity_log[-5:]:  # Show last 5 logs
                                    st.text(f"[{log['timestamp']}] {log['agent']}: {log['message']}")
                    
                    elif stream_chunk['type'] == 'analysis_chunk':
                        # Stream the analysis content as it's generated
                        chunk_content = stream_chunk['content']
                        full_content = stream_chunk['full_content']
                        
                        # Update the conversation history with streaming content
                        idx = len(st.session_state.conversation_history) - 1
                        st.session_state.conversation_history[idx]['content'] = full_content
                        
                        # Update the response area with current content
                        with response_area.container():
                            st.markdown("**Legal Assistant** *(Streaming...)*")
                            st.info(full_content)
                            st.caption("🔄 Analysis is being generated in real-time...")
                    
                    elif stream_chunk['type'] == 'workflow_complete':
                        # Final result
                        result = stream_chunk['result']
                        
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
                            
                            # Clear streaming display and show final result
                            response_area.empty()
                            st.success("✅ **Analysis Complete!** Your legal research is ready.")
                        else:
                            idx = len(st.session_state.conversation_history) - 1
                            st.session_state.conversation_history[idx]['content'] = "I apologize, but I was unable to generate a response for your query."
                            response_area.empty()
                            st.error("❌ Unable to generate response. Please try again.")
                        
                        break
                    
                    elif stream_chunk['type'] == 'error':
                        # Handle errors
                        error_msg = stream_chunk['message']
                        add_activity_log("system", f"Error in processing: {error_msg}", "error")
                        
                        idx = len(st.session_state.conversation_history) - 1
                        st.session_state.conversation_history[idx]['content'] = f"I apologize, but I encountered an error: {error_msg}"
                        
                        response_area.empty()
                        st.error(f"❌ **Error:** {error_msg}")
                        break
                
                st.session_state.processing = False
                
                # Clear the progress area after completion
                progress_area.empty()
                
            except Exception as e:
                add_activity_log("system", f"Error in streaming: {str(e)}", "error")
                update_workflow_status('analysis', 'error', f'Error: {str(e)}')
                
                idx = len(st.session_state.conversation_history) - 1
                st.session_state.conversation_history[idx]['content'] = f"I apologize, but I encountered an error: {str(e)}"
                st.session_state.processing = False
                
                response_area.empty()
                st.error(f"❌ **Error:** {str(e)}")
                # Clear the progress area on error
                progress_area.empty()
            
            # Force a rerun to show final results
            st.rerun()
    
    # Chat input area
    st.markdown("---")
    
    # Check if system is ready
    system_ready = st.session_state.vector_store_initialized and st.session_state.llm_initialized
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        query_input = st.text_input(
            "Ask your legal question:",
            key="query_input",
            placeholder="Example: What are the requirements for a valid contract?" if system_ready else "System not ready - please check status above",
            disabled=st.session_state.processing or not system_ready
        )
    
    with col2:
        submit_button = st.button(
            "Send" if not st.session_state.processing else "Processing...",
            key="submit", 
            disabled=st.session_state.processing or not system_ready
        )
    
    # Show system status if not ready
    if not system_ready:
        if not st.session_state.vector_store_initialized:
            st.warning("⚠️ **Database not ready** - Please check the database status in the sidebar.")
        if not st.session_state.llm_initialized:
            st.warning("⚠️ **LLM API not ready** - Please check the LLM status in the sidebar and ensure LM Studio is running with a model loaded.")
    
    # Process query when submitted
    if submit_button and query_input.strip() and not st.session_state.processing and system_ready:
        process_new_query(query_input.strip())
        st.rerun()

if __name__ == "__main__":
    main()
