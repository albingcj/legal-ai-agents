from typing import Dict, Any
from .base_agent import BaseAgent
import re

class CoordinatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Coordinator",
            description="Orchestrates the legal consultation workflow"
        )
        
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate user query, prepare for research"""
        self.log("Starting coordination process")
        
        # Get user query
        user_query = state.get('query', '')
        if not user_query:
            state['error'] = "No query provided"
            return state
        
        # Parse query and extract legal context
        parsed_query = self.parse_legal_query(user_query)
        
        # Update state
        state.update({
            'original_query': user_query,
            'parsed_query': parsed_query,
            'legal_domain': parsed_query.get('domain', 'general'),
            'query_type': parsed_query.get('type', 'general'),
            'status': 'coordinated'
        })
        
        self.log(f"Coordinated query: {parsed_query['cleaned_query']}")
        self.log(f"Legal domain: {parsed_query.get('domain', 'general')}")
        
        return state
    
    def parse_legal_query(self, query: str) -> Dict[str, Any]:
        """Parse legal query and extract relevant information"""
        query_lower = query.lower()
        
        # Identify legal domain
        domain_keywords = {
            'contract_law': ['contract', 'agreement', 'breach', 'consideration', 'offer', 'acceptance'],
            'employment_law': ['employment', 'job', 'termination', 'at-will', 'discrimination', 'harassment'],
            'intellectual_property': ['copyright', 'patent', 'trademark', 'fair use', 'infringement'],
            'criminal_law': ['criminal', 'miranda', 'rights', 'arrest', 'charges', 'defense'],
            'tort_law': ['injury', 'negligence', 'liability', 'damages', 'accident', 'malpractice']
        }
        
        detected_domain = 'general'
        max_matches = 0
        
        for domain, keywords in domain_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if matches > max_matches:
                max_matches = matches
                detected_domain = domain
        
        # Identify query type
        query_type = 'general'
        if any(word in query_lower for word in ['what', 'how', 'when', 'where', 'why']):
            query_type = 'information'
        elif any(word in query_lower for word in ['should', 'can', 'may', 'advice']):
            query_type = 'advice'
        elif any(word in query_lower for word in ['requirements', 'steps', 'process']):
            query_type = 'procedure'
        
        # Clean query for better search
        cleaned_query = re.sub(r'[^\w\s]', ' ', query).strip()
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query)
        
        return {
            'cleaned_query': cleaned_query,
            'domain': detected_domain,
            'type': query_type,
            'keywords': [word for word in cleaned_query.split() if len(word) > 2],
            'confidence': max_matches / len(domain_keywords.get(detected_domain, []))
        }

# Test the coordinator
if __name__ == "__main__":
    coordinator = CoordinatorAgent()
    
    test_queries = [
        "What are the requirements for a valid contract?",
        "Can my employer terminate me without cause?",
        "How does copyright fair use work?",
        "What are Miranda rights?"
    ]
    
    for query in test_queries:
        state = {'query': query}
        result = coordinator.process(state)
        print(f"\nQuery: {query}")
        print(f"Domain: {result['legal_domain']}")
        print(f"Type: {result['query_type']}")