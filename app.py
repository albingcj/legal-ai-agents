import streamlit as st
import sys
import os
from typing import Dict, Any
import time
from pathlib import Path

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import the workflow and tools
from workflows.legal_workflow import execute_legal_workflow

@st.cache_resource
def initialize_database():
    """Initialize the legal document database on first run"""
    try:
        from tools.document_loader import LegalDocumentLoader
        from tools.vector_store import LegalVectorStore
        
        # Check if database already exists and has data
        chroma_db_path = Path("./chroma_db")
        if chroma_db_path.exists():
            try:
                vector_store = LegalVectorStore()
                # Try to get collection info to see if it has data
                collection_count = vector_store.get_collection_stats().get('total_documents', 0)
                if collection_count > 0:
                    return f"✅ Database already initialized with {collection_count} documents"
            except:
                pass  # If error, continue with initialization
        
        # Initialize database with sample documents
        st.info("🔄 Initializing legal document database... This may take a moment.")
        
        loader = LegalDocumentLoader()
        vector_store = LegalVectorStore()
        
        prepared_docs = loader.prepare_documents_for_vectorstore()
        documents = [doc['text'] for doc in prepared_docs]
        metadata = [doc['metadata'] for doc in prepared_docs]
        
        vector_store.add_documents(documents, metadata)
        
        return f"✅ Database initialized successfully with {len(documents)} legal documents!"
        
    except Exception as e:
        return f"❌ Database initialization failed: {str(e)}"

def main():
    st.set_page_config(
        page_title="Legal AI Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚖️ Legal AI Assistant")
    st.markdown("Get intelligent legal analysis and research assistance")
    
    # Initialize database on startup
    with st.spinner("Setting up legal database..."):
        db_status = initialize_database()
    
    # Show database status
    if "✅" in db_status:
        st.success(db_status)
    elif "❌" in db_status:
        st.error(db_status)
        st.warning("Some features may not work properly. Please check your setup.")
    else:
        st.info(db_status)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        timeout = st.slider("Query Timeout (seconds)", 60, 600, 300)
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This AI assistant helps with legal research and analysis.
        
        **Supported Areas:**
        - Contract Law
        - Employment Law
        - Intellectual Property
        - Criminal Law
        - Tort Law
        """)
        
        st.markdown("---")
        st.warning("**Disclaimer:** This tool provides general information only. Always consult with a qualified attorney for legal advice.")
    
    # Main interface
    st.markdown("## Ask Your Legal Question")
    
    # Query input
    query = st.text_area(
        "Enter your legal question:",
        height=100,
        placeholder="e.g., What are the requirements for a valid contract in employment agreements?"
    )
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        submit_button = st.button("🔍 Analyze", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ Clear")
    
    if clear_button:
        st.rerun()
    
    # Process query
    if submit_button and query.strip():
        with st.spinner("Processing your legal query..."):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress updates
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 20:
                    status_text.text("🎯 Analyzing query...")
                elif i < 60:
                    status_text.text("🔍 Researching legal documents...")
                elif i < 90:
                    status_text.text("📋 Generating analysis...")
                else:
                    status_text.text("✅ Finalizing response...")
                time.sleep(0.02)  # Small delay for visual effect
            
            # Execute workflow
            try:
                result = execute_legal_workflow(query, timeout=timeout)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                display_results(result)
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"An error occurred: {str(e)}")
                st.markdown("Please try again with a simpler question or contact support.")
    
    elif submit_button and not query.strip():
        st.warning("Please enter a legal question to analyze.")

def display_results(result: Dict[str, Any]):
    """Display the workflow results in a user-friendly format"""
    
    # Status indicator
    status = result.get('workflow_status', 'unknown')
    if status == 'completed':
        st.success("✅ Analysis completed successfully!")
    elif status == 'completed_with_error':
        st.warning("⚠️ Analysis completed with issues")
    else:
        st.error("❌ Analysis failed")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        processing_time = result.get('processing_time', 0)
        st.metric("Processing Time", f"{processing_time:.1f}s")
    
    with col2:
        confidence = result.get('confidence_score', 0)
        st.metric("Confidence Score", f"{confidence:.2f}")
    
    with col3:
        final_response = result.get('final_response', {})
        citations = final_response.get('citation_count', 0)
        st.metric("Citations", citations)
    
    with col4:
        legal_domain = result.get('legal_domain', 'Unknown')
        st.metric("Legal Domain", legal_domain)
    
    # Main analysis
    if final_response:
        st.markdown("## 📋 Legal Analysis")
        
        # Analysis text
        analysis = final_response.get('analysis', 'No analysis available')
        st.markdown(analysis)
        
        # Key points
        key_points = final_response.get('key_points', [])
        if key_points:
            st.markdown("### 🔑 Key Points")
            for point in key_points:
                st.markdown(f"• {point}")
        
        # Sources
        sources = final_response.get('sources', [])
        if sources:
            st.markdown("### 📚 Sources")
            with st.expander(f"View {len(sources)} sources"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**{i}.** {source}")
        
        # Disclaimer
        disclaimer = final_response.get('disclaimer', '')
        if disclaimer:
            st.markdown("### ⚠️ Legal Disclaimer")
            st.info(disclaimer)
    
    # Workflow details (expandable)
    with st.expander("🔧 Technical Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Workflow Information:**")
            st.json({
                'workflow_id': result.get('workflow_id', 'N/A'),
                'timestamp': result.get('timestamp', 'N/A'),
                'status': result.get('workflow_status', 'N/A'),
                'query_type': result.get('query_type', 'N/A')
            })
        
        with col2:
            agent_logs = result.get('agent_logs', [])
            if agent_logs:
                st.markdown("**Agent Execution Log:**")
                for log in agent_logs:
                    st.markdown(f"• **{log.get('agent', 'Unknown')}**: {log.get('status', 'Unknown')}")
    
    # Error details if any
    if result.get('error'):
        st.markdown("### 🚨 Error Details")
        st.error(result['error'])

if __name__ == "__main__":
    main()
