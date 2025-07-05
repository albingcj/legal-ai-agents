import sys
import os
import time
import unittest
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.legal_workflow import execute_legal_workflow
from agents.research_agent import ResearchAgent
from agents.analysis_agent import AnalysisAgent
from tools.document_loader import LegalDocumentLoader
from tools.vector_store import LegalVectorStore

class TestCompleteSystem(unittest.TestCase):
    """Comprehensive system tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        print("Setting up test environment...")
        
        # Initialize components
        cls.loader = LegalDocumentLoader()
        cls.vector_store = LegalVectorStore()
        
        # Load documents
        prepared_docs = cls.loader.prepare_documents_for_vectorstore()
        documents = [doc['text'] for doc in prepared_docs]
        metadata = [doc['metadata'] for doc in prepared_docs]
        cls.vector_store.add_documents(documents, metadata)
        
        print("Test environment ready!")
    
    def test_workflow_execution(self):
        """Test complete workflow execution"""
        test_queries = [
            "What are the requirements for a valid contract?",
            "Can my employer terminate me without cause?",
            "How does copyright fair use work?",
            "What are Miranda rights?",
            "What is negligence in tort law?"
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                print(f"\nTesting: {query}")
                
                start_time = time.time()
                result = execute_legal_workflow(query)
                processing_time = time.time() - start_time
                
                # Basic assertions
                self.assertIsNotNone(result)
                self.assertIn('workflow_status', result)
                self.assertIn('final_response', result)
                
                # Performance assertion
                self.assertLess(processing_time, 60, "Workflow took too long")
                
                # Quality assertions
                if result.get('final_response'):
                    response = result['final_response']
                    self.assertIn('analysis', response)
                    self.assertGreater(len(response['analysis']), 100, "Analysis too short")
                    self.assertIn('disclaimer', response)
                
                print(f"   ✅ Completed in {processing_time:.2f}s")
                print(f"   Status: {result.get('workflow_status')}")
                print(f"   Confidence: {result.get('confidence_score', 0):.2f}")
    
    def test_error_handling(self):
        """Test error handling capabilities"""
        error_cases = [
            "",  # Empty query
            "x",  # Too short
            "a" * 2000,  # Too long
        ]
        
        for query in error_cases:
            with self.subTest(query=f"Error case: {len(query)} chars"):
                result = execute_legal_workflow(query)
                
                # Should handle errors gracefully
                self.assertIsNotNone(result)
                self.assertIn('workflow_status', result)
                
                # Should provide fallback response
                if result.get('final_response'):
                    response = result['final_response']
                    self.assertIn('analysis', response)
    
    def test_domain_detection(self):
        """Test legal domain detection accuracy"""
        domain_tests = [
            ("What are contract requirements?", "contract_law"),
            ("Employment termination rights", "employment_law"),
            ("Copyright fair use guidelines", "intellectual_property"),
            ("Miranda rights explanation", "criminal_law"),
            ("Negligence liability standards", "tort_law")
        ]
        
        for query, expected_domain in domain_tests:
            with self.subTest(query=query, expected=expected_domain):
                result = execute_legal_workflow(query)
                
                detected_domain = result.get('legal_domain', '').lower().replace(' ', '_')
                self.assertEqual(detected_domain, expected_domain,
                               f"Expected {expected_domain}, got {detected_domain}")
    
    def test_research_quality(self):
        """Test research result quality"""
        query = "What are the requirements for a valid contract?"
        
        research_agent = ResearchAgent()
        state = {
            'query': query,
            'parsed_query': {
                'cleaned_query': 'requirements valid contract',
                'domain': 'contract_law',
                'type': 'information'
            },
            'legal_domain': 'contract_law'
        }
        
        result = research_agent.process(state)
        
        self.assertIn('research_results', result)
        self.assertGreater(len(result['research_results']), 0)
        
        # Check result quality
        for res in result['research_results'][:3]:
            self.assertIn('composite_score', res)
            self.assertGreater(res['composite_score'], 0.3, "Low relevance score")
            self.assertIn('document', res)
            self.assertGreater(len(res['document']), 50, "Document too short")
    
    def test_analysis_quality(self):
        """Test analysis generation quality"""
        # Mock research results
        mock_research_results = [
            {
                'document': 'A contract requires offer, acceptance, consideration, and mutual assent.',
                'metadata': {'title': 'Contract Law Basics', 'domain': 'contract_law'},
                'composite_score': 0.9
            }
        ]
        
        analysis_agent = AnalysisAgent()
        state = {
            'query': 'What are contract requirements?',
            'research_results': mock_research_results,
            'legal_domain': 'contract_law'
        }
        
        result = analysis_agent.process(state)
        
        self.assertIn('legal_analysis', result)
        self.assertIn('final_response', result)
        
        analysis = result['legal_analysis']
        self.assertGreater(len(analysis), 200, "Analysis too short")
        self.assertIn('contract', analysis.lower())
        
        # Check final response structure
        final_response = result['final_response']
        required_keys = ['analysis', 'key_points', 'legal_domain', 'disclaimer']
        for key in required_keys:
            self.assertIn(key, final_response, f"Missing key: {key}")

def run_performance_benchmark():
    """Run performance benchmark"""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    test_queries = [
        "What are the requirements for a valid contract?",
        "Can my employer terminate me without cause?",
        "How does copyright fair use work?"
    ]
    
    total_time = 0
    successful_runs = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nBenchmark {i}/3: {query}")
        
        start_time = time.time()
        try:
            result = execute_legal_workflow(query)
            processing_time = time.time() - start_time
            total_time += processing_time
            
            if result.get('workflow_status') == 'completed':
                successful_runs += 1
                confidence = result.get('confidence_score', 0)
                print(f"   ✅ Success: {processing_time:.2f}s, Confidence: {confidence:.2f}")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"   ❌ Error: {str(e)} ({processing_time:.2f}s)")
    
    # Results
    print(f"\n" + "-"*30)
    print(f"BENCHMARK RESULTS:")
    print(f"Successful runs: {successful_runs}/{len(test_queries)}")
    print(f"Average time: {total_time/len(test_queries):.2f}s")
    print(f"Success rate: {successful_runs/len(test_queries)*100:.1f}%")

def main():
    """Run all tests"""
    print("Starting Complete System Tests")
    print("="*50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance benchmark
    run_performance_benchmark()
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)

if __name__ == "__main__":
    main()
