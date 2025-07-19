import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.llm_client import UniversalLLMClient
from tools.vector_store import LegalVectorStore
from tools.document_loader import LegalDocumentLoader
from agents.coordinator import CoordinatorAgent

def test_llm_client():
    """Test LLM client connection"""
    print("=== Testing Universal LLM Client ===")
    client = UniversalLLMClient()
    
    # Test health check first
    health = client.health_check()
    print(f"Health Status: {health}")
    
    if health['status'] != 'healthy':
        print(f"LLM not healthy: {health['message']}")
        return False
    
    messages = [
        {"role": "user", "content": "Hello! Are you working properly?"}
    ]
    
    response = client.generate_response(messages)
    print(f"Response: {response}")
    return "error" not in response.lower()

def test_vector_store():
    """Test vector store operations"""
    print("\n=== Testing Vector Store ===")
    vs = LegalVectorStore()
    
    # Load sample documents
    loader = LegalDocumentLoader()
    prepared_docs = loader.prepare_documents_for_vectorstore()
    
    # Add documents
    documents = [doc['text'] for doc in prepared_docs]
    metadata = [doc['metadata'] for doc in prepared_docs]
    
    vs.add_documents(documents, metadata)
    
    # Test search
    results = vs.search("What is a contract?", top_k=3)
    
    print(f"Search results for 'What is a contract?':")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['relevance_score']:.3f}")
        print(f"   Domain: {result['metadata']['domain']}")
        print(f"   Text: {result['document'][:100]}...")
    
    stats = vs.get_collection_stats()
    print(f"\nCollection stats: {stats}")
    
    return len(results) > 0

def test_coordinator():
    """Test coordinator agent"""
    print("\n=== Testing Coordinator Agent ===")
    coordinator = CoordinatorAgent()
    
    test_query = "What are the requirements for a valid contract?"
    state = {'query': test_query}
    
    result = coordinator.process(state)
    
    print(f"Original query: {result['original_query']}")
    print(f"Parsed query: {result['parsed_query']['cleaned_query']}")
    print(f"Legal domain: {result['legal_domain']}")
    print(f"Query type: {result['query_type']}")
    
    return result['status'] == 'coordinated'

def test_full_pipeline():
    """Test the complete pipeline"""
    print("\n=== Testing Full Pipeline ===")
    
    # Initialize components
    llm_client = UniversalLLMClient()
    vs = LegalVectorStore()
    coordinator = CoordinatorAgent()
    
    # Test query
    query = "What are the requirements for a valid contract?"
    
    # Step 1: Coordinate
    state = {'query': query}
    state = coordinator.process(state)
    
    # Step 2: Search
    search_results = vs.search(state['parsed_query']['cleaned_query'], top_k=3)
    
    # Step 3: Generate response
    context = "\n\n".join([result['document'] for result in search_results])
    
    messages = [
        {"role": "system", "content": "You are a legal assistant. Provide helpful legal information based on the context provided. Always include disclaimers that this is not legal advice."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nPlease provide a helpful response based on the context."}
    ]
    
    response = llm_client.generate_response(messages)
    
    print(f"Query: {query}")
    print(f"Domain: {state['legal_domain']}")
    print(f"Retrieved {len(search_results)} relevant documents")
    print(f"Response: {response[:200]}...")
    
    return True

if __name__ == "__main__":
    print("Starting Day 1 Integration Tests\n")
    
    # Run tests
    tests = [
        ("LM Studio Client", test_llm_client),
        ("Vector Store", test_vector_store),
        ("Coordinator Agent", test_coordinator),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
            print(f"✅ {test_name}: {'PASSED' if results[test_name] else 'FAILED'}")
        except Exception as e:
            results[test_name] = False
            print(f"❌ {test_name}: ERROR - {str(e)}")
    
    print(f"\n=== Day 1 Test Summary ===")
    passed = sum(results.values())
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Ready for Day 2!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")