from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from agents.coordinator import CoordinatorAgent
from agents.research_agent import ResearchAgent
from agents.analysis_agent import AnalysisAgent
import logging
import threading
import time
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as ConcurrentTimeoutError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalWorkflowState(TypedDict):
    """Complete state definition for legal workflow"""
    # Input
    query: str
    
    # Coordinator outputs
    parsed_query: Optional[Dict[str, Any]]
    legal_domain: Optional[str]
    query_type: Optional[str]
    
    # Research outputs
    research_results: Optional[List[Dict]]
    enhanced_queries: Optional[List[str]]
    key_concepts: Optional[List[str]]
    research_metadata: Optional[Dict[str, Any]]
    research_status: Optional[str]
    
    # Analysis outputs
    legal_analysis: Optional[str]
    analysis_with_citations: Optional[str]
    final_response: Optional[Dict[str, Any]]
    confidence_score: Optional[float]
    analysis_metadata: Optional[Dict[str, Any]]
    analysis_status: Optional[str]
    
    # Workflow management
    current_step: Optional[str]
    error: Optional[str]
    processing_time: Optional[float]
    workflow_status: Optional[str]
    
    # Metadata
    workflow_id: Optional[str]
    timestamp: Optional[str]
    agent_logs: Optional[List[Dict]]

class ThreadSafeLegalWorkflow:
    """Thread-safe workflow executor with retry mechanisms"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._active_workflows = {}
        self.max_retries = 3
        self.retry_delay = 1  # seconds
    
    def execute_with_retry(self, query: str, timeout: int = 300) -> Dict[str, Any]:
        """Execute workflow with retry mechanism and timeout"""
        workflow_id = str(uuid.uuid4())
        last_error = None
        
        with self._lock:
            if workflow_id in self._active_workflows:
                raise ValueError("Workflow already running")
            self._active_workflows[workflow_id] = 'running'
        
        try:
            for attempt in range(self.max_retries):
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(self._execute_workflow_internal, query, workflow_id)
                        result = future.result(timeout=timeout)
                        
                        with self._lock:
                            self._active_workflows[workflow_id] = 'completed'
                        
                        return result
                        
                except ConnectionError as e:
                    last_error = e
                    logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    
                except ConcurrentTimeoutError as e:
                    last_error = e
                    logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
                    # Don't retry timeouts immediately
                    break
                    
                except Exception as e:
                    last_error = e
                    logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        break
            
            # All retries failed
            with self._lock:
                self._active_workflows[workflow_id] = 'failed'
            
            return self._create_fallback_response(query, last_error)
            
        except Exception as e:
            with self._lock:
                self._active_workflows[workflow_id] = 'failed'
            logger.error(f"Critical workflow error: {e}")
            return self._create_fallback_response(query, e)
    
    def _execute_workflow_internal(self, query: str, workflow_id: str) -> Dict[str, Any]:
        """Internal workflow execution"""
        workflow = create_legal_workflow()
        
        initial_state = {
            'query': query,
            'workflow_id': workflow_id,
            'timestamp': datetime.now().isoformat(),
            'workflow_status': 'started'
        }
        
        return workflow.invoke(initial_state)
    
    def _detect_domain_from_query(self, query: str) -> Optional[str]:
        """Detect legal domain from query text"""
        domain_keywords = {
            'contract_law': ['contract', 'agreement', 'breach', 'consideration', 'offer', 'acceptance'],
            'employment_law': ['employment', 'job', 'termination', 'at-will', 'discrimination', 'harassment'],
            'intellectual_property': ['copyright', 'patent', 'trademark', 'fair use', 'infringement'],
            'criminal_law': ['criminal', 'miranda', 'rights', 'arrest', 'charges', 'defense'],
            'tort_law': ['injury', 'negligence', 'liability', 'damages', 'accident', 'malpractice']
        }
        
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return 'general'
    
    def _create_fallback_response(self, query: str, error: Exception) -> Dict[str, Any]:
        """Create intelligent fallback response"""
        domain = self._detect_domain_from_query(query)
        
        fallback_guidance = {
            'contract_law': "For contract-related questions, consider consulting contract law resources or an attorney specializing in contract law.",
            'employment_law': "For employment issues, contact your HR department or an employment attorney for specific guidance.",
            'intellectual_property': "For IP matters, consult with an intellectual property attorney who can provide specialized advice.",
            'criminal_law': "For criminal law questions, contact a criminal defense attorney immediately for urgent matters.",
            'tort_law': "For personal injury or tort claims, consult with a personal injury attorney for case evaluation."
        }
        
        guidance = fallback_guidance.get(domain, "Please consult with a qualified attorney for your specific legal question.")
        
        return {
            'workflow_status': 'completed_with_error',
            'final_response': {
                'analysis': f"""
I apologize, but I encountered technical difficulties while processing your legal query.

**Your Question:** {query}

**Detected Legal Area:** {domain.replace('_', ' ').title() if domain != 'general' else 'General Law'}

**Immediate Guidance:** {guidance}

**What you can try:**
1. **Rephrase your question** - Try being more specific about your legal issue
2. **Break down complex questions** - Ask about one legal concept at a time
3. **Check system status** - There may be temporary technical issues
4. **Try again later** - System may recover from temporary issues

**For urgent legal matters:** Always consult directly with a qualified attorney in your jurisdiction.

**Technical Details:** {str(error) if error else 'System temporarily unavailable'}
                """,
                'key_points': [
                    "System temporarily unavailable due to technical issues",
                    "Try rephrasing your question or asking about specific legal concepts",
                    "For urgent matters, consult an attorney directly",
                    f"Detected legal area: {domain.replace('_', ' ').title()}"
                ],
                'legal_domain': domain.replace('_', ' ').title() if domain != 'general' else 'General Law',
                'disclaimer': """
**IMPORTANT LEGAL DISCLAIMER:**
This is an error response due to technical difficulties. The information provided is general guidance only.
For legal matters, always consult with a qualified attorney in your jurisdiction who can provide advice
specific to your situation and local laws.
                """,
                'sources': [],
                'citation_count': 0,
                'response_type': 'fallback_response'
            },
            'confidence_score': 0.0,
            'error': str(error) if error else 'System error',
            'processing_time': 0.0,
            'agent_logs': [
                {
                    'agent': 'error_handler',
                    'timestamp': datetime.now().isoformat(),
                    'status': 'completed',
                    'details': {
                        'error_handled': str(error) if error else 'Unknown error',
                        'fallback_provided': True,
                        'domain_detected': domain
                    }
                }
            ]
        }

# Global workflow instance
_workflow_executor = ThreadSafeLegalWorkflow()

def create_legal_workflow() -> StateGraph:
    """Create and configure the legal workflow"""
    
    # Create the workflow graph
    workflow = StateGraph(LegalWorkflowState)
    
    # Add nodes
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("research", research_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # Define edges
    workflow.add_conditional_edges(
        "coordinator",
        coordinator_router,
        {
            "research": "research",
            "error": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "research", 
        research_router,
        {
            "analysis": "analysis",
            "error": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "analysis",
        analysis_router,
        {
            "end": END,
            "error": "error_handler"
        }
    )
    
    workflow.add_edge("error_handler", END)
    
    # Set entry point
    workflow.set_entry_point("coordinator")
    
    return workflow.compile()

def coordinator_node(state: LegalWorkflowState) -> LegalWorkflowState:
    """Process query through coordinator agent"""
    logger.info("Executing coordinator node")
    
    try:
        # Initialize coordinator
        coordinator = CoordinatorAgent()
        
        # Update state
        state['current_step'] = 'coordination'
        state['workflow_status'] = 'processing'
        
        # Process through coordinator
        result_state = coordinator.process(dict(state))
        
        # Merge results back to state
        for key, value in result_state.items():
            state[key] = value
        
        # Add agent logs
        if 'agent_logs' not in state:
            state['agent_logs'] = []
        
        state['agent_logs'].append({
            'agent': 'coordinator',
            'timestamp': datetime.now().isoformat(),
            'status': 'completed' if not state.get('error') else 'failed',
            'details': coordinator.get_agent_info(),
            'processing_time': sum(1 for _ in coordinator.processing_history) * 0.1  # Estimate
        })
        
        logger.info("Coordinator node completed successfully")
        
    except Exception as e:
        logger.error(f"Coordinator node error: {str(e)}")
        state['error'] = f"Coordinator failed: {str(e)}"
        state['workflow_status'] = 'error'
        
        # Add error log
        if 'agent_logs' not in state:
            state['agent_logs'] = []
        
        state['agent_logs'].append({
            'agent': 'coordinator',
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': str(e),
            'details': {'error_type': 'coordinator_failure'}
        })
    
    return state

def research_node(state: LegalWorkflowState) -> LegalWorkflowState:
    """Process query through research agent"""
    logger.info("Executing research node")
    
    try:
        # Initialize research agent
        research_agent = ResearchAgent()
        
        # Update state
        state['current_step'] = 'research'
        
        # Process through research agent
        result_state = research_agent.process(dict(state))
        
        # Merge results back to state
        for key, value in result_state.items():
            state[key] = value
        
        # Add agent logs
        if 'agent_logs' not in state:
            state['agent_logs'] = []
        
        state['agent_logs'].append({
            'agent': 'research',
            'timestamp': datetime.now().isoformat(),
            'status': 'completed' if state.get('research_status') == 'completed' else 'failed',
            'details': research_agent.get_agent_info(),
            'statistics': research_agent.get_search_statistics(),
            'results_found': len(state.get('research_results', []))
        })
        
        logger.info("Research node completed successfully")
        
    except Exception as e:
        logger.error(f"Research node error: {str(e)}")
        state['error'] = f"Research failed: {str(e)}"
        state['research_status'] = 'failed'
        state['workflow_status'] = 'error'
        
        # Add error log
        if 'agent_logs' not in state:
            state['agent_logs'] = []
        
        state['agent_logs'].append({
            'agent': 'research',
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': str(e),
            'details': {'error_type': 'research_failure'}
        })
    
    return state

def analysis_node(state: LegalWorkflowState) -> LegalWorkflowState:
    """Process results through analysis agent"""
    logger.info("Executing analysis node")
    
    try:
        # Initialize analysis agent
        analysis_agent = AnalysisAgent()
        
        # Update state
        state['current_step'] = 'analysis'
        
        # Process through analysis agent
        result_state = analysis_agent.process(dict(state))
        
        # Merge results back to state
        for key, value in result_state.items():
            state[key] = value
        
        # Add agent logs
        if 'agent_logs' not in state:
            state['agent_logs'] = []
        
        state['agent_logs'].append({
            'agent': 'analysis',
            'timestamp': datetime.now().isoformat(),
            'status': 'completed' if state.get('analysis_status') == 'completed' else 'failed',
            'details': analysis_agent.get_agent_info(),
            'analysis_length': len(state.get('legal_analysis', '')),
            'confidence_score': state.get('confidence_score', 0.0)
        })
        
        # Mark workflow as completed
        if state.get('analysis_status') == 'completed':
            state['workflow_status'] = 'completed'
        
        logger.info("Analysis node completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis node error: {str(e)}")
        state['error'] = f"Analysis failed: {str(e)}"
        state['analysis_status'] = 'failed'
        state['workflow_status'] = 'error'
        
        # Add error log
        if 'agent_logs' not in state:
            state['agent_logs'] = []
        
        state['agent_logs'].append({
            'agent': 'analysis',
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': str(e),
            'details': {'error_type': 'analysis_failure'}
        })
    
    return state

def error_handler_node(state: LegalWorkflowState) -> LegalWorkflowState:
    """Handle errors and provide fallback response"""
    logger.info("Executing error handler node")
    
    try:
        error_message = state.get('error', 'Unknown error occurred')
        current_step = state.get('current_step', 'unknown')
        query = state.get('query', '')
        
        # Detect domain for better guidance
        domain_keywords = {
            'contract_law': ['contract', 'agreement', 'breach', 'consideration'],
            'employment_law': ['employment', 'job', 'termination', 'at-will'],
            'intellectual_property': ['copyright', 'patent', 'trademark', 'fair use'],
            'criminal_law': ['criminal', 'miranda', 'rights', 'arrest'],
            'tort_law': ['injury', 'negligence', 'liability', 'damages']
        }
        
        detected_domain = 'general'
        query_lower = query.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_domain = domain
                break
        
        # Create fallback response
        fallback_response = {
            'analysis': f"""
I apologize, but I encountered an error while processing your legal query.

**Error Details:**
- Processing Stage: {current_step.title()}
- Issue: {error_message}
- Query: {query}

**Detected Legal Area:** {detected_domain.replace('_', ' ').title()}

**What you can try:**
1. **Simplify your question** - Break complex queries into smaller, specific questions
2. **Use clear legal terms** - Include relevant legal terminology in your query
3. **Check spelling** - Ensure legal terms are spelled correctly
4. **Try again** - System issues may be temporary

**Supported Legal Areas:**
- Contract Law (agreements, breach, consideration)
- Employment Law (termination, discrimination, workplace rights)
- Intellectual Property (copyright, patents, trademarks)
- Criminal Law (rights, procedures, defenses)
- Tort Law (negligence, liability, personal injury)

**For urgent legal matters:** Please consult with a qualified attorney in your jurisdiction who can provide immediate assistance.
            """,
            'key_points': [
                f"Error occurred during {current_step} stage",
                "Try simplifying or rephrasing your question",
                "System may have temporary issues",
                "Consult an attorney for urgent legal matters",
                f"Detected area: {detected_domain.replace('_', ' ').title()}"
            ],
            'legal_domain': detected_domain.replace('_', ' ').title(),
            'disclaimer': """
**IMPORTANT:** This is an error response. The information provided is general guidance only.
For legal matters, always consult with a qualified attorney in your jurisdiction.
            """,
            'sources': [],
            'citation_count': 0,
            'response_type': 'error_response'
        }
        
        # Update state
        state['final_response'] = fallback_response
        state['confidence_score'] = 0.0
        state['workflow_status'] = 'completed_with_error'
        
        # Add error log
        if 'agent_logs' not in state:
            state['agent_logs'] = []
        
        state['agent_logs'].append({
            'agent': 'error_handler',
            'timestamp': datetime.now().isoformat(),
            'status': 'completed',
            'details': {
                'error_handled': error_message,
                'fallback_provided': True,
                'detected_domain': detected_domain,
                'failed_stage': current_step
            }
        })
        
        logger.info("Error handler completed - fallback response provided")
        
    except Exception as e:
        logger.error(f"Error handler failed: {str(e)}")
        # Final fallback
        state['final_response'] = {
            'analysis': 'A critical system error occurred. Please try again later or contact support.',
            'key_points': ['Critical system error occurred'],
            'legal_domain': 'System Error',
            'disclaimer': 'System error - please retry or contact support',
            'sources': [],
            'citation_count': 0,
            'response_type': 'critical_error'
        }
        state['workflow_status'] = 'critical_error'
        state['confidence_score'] = 0.0
    
    return state

# Router functions
def coordinator_router(state: LegalWorkflowState) -> str:
    """Route after coordinator processing"""
    if state.get('error'):
        return "error"
    return "research"

def research_router(state: LegalWorkflowState) -> str:
    """Route after research processing"""
    if state.get('error') or state.get('research_status') == 'failed':
        return "error"
    return "analysis"

def analysis_router(state: LegalWorkflowState) -> str:
    """Route after analysis processing"""
    if state.get('error') or state.get('analysis_status') == 'failed':
        return "error"
    return "end"

# Public API functions
def execute_legal_workflow(query: str, timeout: int = 300) -> Dict[str, Any]:
    """Execute the complete legal workflow with enhanced error handling"""
    start_time = time.time()
    
    try:
        result = _workflow_executor.execute_with_retry(query, timeout)
        
        # Add processing time
        result['processing_time'] = time.time() - start_time
        
        logger.info(f"Workflow completed in {result['processing_time']:.2f} seconds")
        
        return result
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {str(e)}")
        
        # Return fallback response
        return {
            'query': query,
            'workflow_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'error': f"Workflow execution failed: {str(e)}",
            'workflow_status': 'failed',
            'processing_time': time.time() - start_time,
            'final_response': {
                'analysis': f"""
I apologize, but the system is currently experiencing technical difficulties and cannot process your legal query.

**Your Question:** {query}

**What happened:** {str(e)}

**What you can do:**
1. **Try again in a few minutes** - This may be a temporary issue
2. **Simplify your question** - Try asking about one specific legal concept
3. **Contact support** - If the problem persists
4. **Consult an attorney** - For urgent legal matters

**For immediate legal assistance:** Please contact a qualified attorney in your jurisdiction.
                """,
                'key_points': [
                    "System temporarily unavailable",
                    "Try again later with a simplified question",
                    "Contact attorney for urgent matters"
                ],
                'legal_domain': 'System Error',
                'disclaimer': 'System error - consult attorney for legal advice',
                'sources': [],
                'citation_count': 0,
                'response_type': 'system_error'
            },
            'confidence_score': 0.0
        }

# Test the workflow
if __name__ == "__main__":
    from tools.document_loader import LegalDocumentLoader
    from tools.vector_store import LegalVectorStore
    
    # Setup test environment
    print("Setting up test environment...")
    
    try:
        # Load documents
        loader = LegalDocumentLoader()
        vector_store = LegalVectorStore()
        
        prepared_docs = loader.prepare_documents_for_vectorstore()
        documents = [doc['text'] for doc in prepared_docs]
        metadata = [doc['metadata'] for doc in prepared_docs]
        vector_store.add_documents(documents, metadata)
        
        print("Documents loaded successfully!")
        
        # Test workflow
        test_queries = [
            "What are the requirements for a valid contract?",
            "Can my employer terminate me without cause?",
            "How does copyright fair use work?"
        ]
        
        for query in test_queries:
            print(f"\nTesting workflow with query: {query}")
            
            result = execute_legal_workflow(query)
            
            print(f"=== Results for: {query} ===")
            print(f"Status: {result.get('workflow_status')}")
            print(f"Processing time: {result.get('processing_time', 0):.2f} seconds")
            print(f"Legal domain: {result.get('legal_domain')}")
            print(f"Confidence score: {result.get('confidence_score', 0):.2f}")
            
            if result.get('final_response'):
                response = result['final_response']
                print(f"Analysis preview: {response['analysis'][:200]}...")
                print(f"Key points: {len(response.get('key_points', []))}")
                print(f"Sources: {len(response.get('sources', []))}")
            
            if result.get('error'):
                print(f"Error: {result['error']}")
            
            print("-" * 50)
    
    except Exception as e:
        print(f"Test failed: {str(e)}")
