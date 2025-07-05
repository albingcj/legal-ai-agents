import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.research_agent import ResearchAgent
from tools.document_loader import LegalDocumentLoader
import time

def test_research_agent_comprehensive():
    """Comprehensive test of the research agent"""
    print("=== Comprehensive Research Agent Test ===\n")
    
    # Initialize components
    research_agent = ResearchAgent()
    loader = LegalDocumentLoader()
    
    # Load documents
    print("1. Loading documents into vector store...")
    prepared_docs = loader.prepare_documents_for_vectorstore()
    documents = [doc['text'] for doc in prepared_docs]
    metadata = [doc['metadata'] for doc in prepared_docs]
    research_agent.vector_store.add_documents(documents, metadata)
    print(f"   Loaded {len(documents)} document chunks")
    
    # Test queries for different domains
    test_cases = [
        {
            'name': 'Contract Law Query',
            'query': 'What are the requirements for a valid contract?',
            'domain': 'contract_law',
            'expected_concepts': ['contract', 'agreement', 'consideration']
        },
        {
            'name': 'Employment Law Query',
            'query': 'Can my employer terminate me without cause?',
            'domain': 'employment_law',
            'expected_concepts': ['employment', 'termination', 'at-will']
        },
        {
            'name': 'IP Law Query',
            'query': 'How does copyright fair use work?',
            'domain': 'intellectual_property',
            'expected_concepts': ['copyright', 'fair', 'use']
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing {test_case['name']}")
        print(f"   Query: {test_case['query']}")
        
        # Prepare state
        state = {
            'query': test_case['query'],
            'parsed_query': {
                'cleaned_query': test_case['query'].lower(),
                'domain': test_case['domain'],
                'type': 'information',
                'keywords': test_case['query'].lower().split()
            },
            'legal_domain': test_case['domain']
        }
        
        # Process query
        start_time = time.time()
        result_state = research_agent.process(state)
        processing_time = time.time() - start_time
        
        # Analyze results
        if 'research_results' in result_state:
            research_results = result_state['research_results']
            key_concepts = result_state.get('key_concepts', [])
            metadata = result_state.get('research_metadata', {})
            
            print(f"   ✅ Success! Found {len(research_results)} results")
            print(f"   ⏱️  Processing time: {processing_time:.2f} seconds")
            print(f"   🔍 Key concepts: {key_concepts[:5]}")
            print(f"   📊 Avg relevance: {metadata.get('avg_relevance_score', 0):.3f}")
            
            # Check if expected concepts are found
            found_concepts = sum(1 for concept in test_case['expected_concepts'] 
                               if any(concept in kc.lower() for kc in key_concepts))
            concept_coverage = found_concepts / len(test_case['expected_concepts'])
            print(f"   🎯 Concept coverage: {concept_coverage:.1%}")
            
            # Show top result
            if research_results:
                top_result = research_results[0]
                print(f"   🏆 Top result score: {top_result['composite_score']:.3f}")
                print(f"      Domain: {top_result['metadata']['domain']}")
                print(f"      Preview: {top_result['document'][:100]}...")
            
            results.append({
                'test_case': test_case['name'],
                'success': True,
                'processing_time': processing_time,
                'result_count': len(research_results),
                'concept_coverage': concept_coverage,
                'avg_relevance': metadata.get('avg_relevance_score', 0)
            })
        else:
            print(f"   ❌ Failed: {result_state.get('error', 'Unknown error')}")
            results.append({
                'test_case': test_case['name'],
                'success': False,
                'error': result_state.get('error', 'Unknown error')
            })
    
    # Summary
    print(f"\n=== Test Summary ===")
    successful_tests = sum(1 for r in results if r['success'])
    print(f"Passed: {successful_tests}/{len(results)} tests")
    
    if successful_tests > 0:
        avg_time = sum(r['processing_time'] for r in results if r['success']) / successful_tests
        avg_results = sum(r['result_count'] for r in results if r['success']) / successful_tests
        avg_coverage = sum(r['concept_coverage'] for r in results if r['success']) / successful_tests
        
        print(f"Average processing time: {avg_time:.2f} seconds")
        print(f"Average results per query: {avg_results:.1f}")
        print(f"Average concept coverage: {avg_coverage:.1%}")
    
    # Show search statistics
    stats = research_agent.get_search_statistics()
    print(f"\nSearch Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    return successful_tests == len(results)

if __name__ == "__main__":
    success = test_research_agent_comprehensive()
    if success:
        print("\n🎉 All Research Agent tests passed! Ready for Analysis Agent.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
