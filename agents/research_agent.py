from typing import Dict, Any, List, Optional, Tuple
from .base_agent import BaseAgent
from tools.vector_store import LegalVectorStore
import re
from collections import defaultdict
import numpy as np

class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Legal Research Agent",
            description="Advanced RAG-based legal document research with multi-stage search and ranking"
        )
        
        # Initialize vector store
        self.vector_store = LegalVectorStore()
        
        # Legal domain mappings for query enhancement
        self.legal_domains = {
            'contract_law': {
                'synonyms': ['agreement', 'binding', 'consideration', 'breach', 'covenant', 'terms'],
                'key_concepts': ['offer', 'acceptance', 'mutual assent', 'capacity', 'legality'],
                'search_boost': 1.3,
                'required_elements': ['formation', 'performance', 'enforcement']
            },
            'employment_law': {
                'synonyms': ['job', 'work', 'employer', 'employee', 'termination', 'hiring'],
                'key_concepts': ['at-will', 'discrimination', 'harassment', 'wages', 'benefits'],
                'search_boost': 1.2,
                'required_elements': ['workplace', 'labor', 'rights']
            },
            'intellectual_property': {
                'synonyms': ['copyright', 'patent', 'trademark', 'IP', 'infringement', 'fair use'],
                'key_concepts': ['originality', 'protection', 'licensing', 'ownership'],
                'search_boost': 1.4,
                'required_elements': ['creation', 'protection', 'enforcement']
            },
            'criminal_law': {
                'synonyms': ['crime', 'criminal', 'defendant', 'prosecution', 'charges'],
                'key_concepts': ['mens rea', 'actus reus', 'miranda', 'rights', 'defense'],
                'search_boost': 1.2,
                'required_elements': ['elements', 'defenses', 'procedure']
            },
            'tort_law': {
                'synonyms': ['injury', 'negligence', 'liability', 'damages', 'harm'],
                'key_concepts': ['duty', 'breach', 'causation', 'harm', 'standard of care'],
                'search_boost': 1.3,
                'required_elements': ['duty', 'breach', 'damages']
            }
        }
        
        # Initialize search statistics
        self.search_stats = {
            'total_searches': 0,
            'avg_results': 0,
            'domain_distribution': defaultdict(int)
        }
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main research pipeline"""
        self.log("Starting legal research process")
        
        # Validate required state
        if not self.validate_state(state, ['parsed_query', 'legal_domain']):
            state['error'] = "Missing required state for research"
            return state
        
        try:
            # Extract query information
            parsed_query = state['parsed_query']
            original_query = parsed_query['cleaned_query']
            legal_domain = state['legal_domain']
            
            self.log(f"Researching query: '{original_query}' in domain: {legal_domain}")
            
            # Step 1: Enhance query with legal terminology
            enhanced_queries = self.enhance_query(original_query, legal_domain)
            self.log(f"Generated {len(enhanced_queries)} enhanced queries")
            
            # Step 2: Perform multi-stage search
            search_results = self.multi_stage_search(enhanced_queries, top_k=15)
            self.log(f"Retrieved {len(search_results)} total results")
            
            # Step 3: Rank and filter results
            ranked_results = self.rank_results(search_results, legal_domain)
            final_results = ranked_results[:8]  # Keep top 8 results
            
            # Step 4: Extract key concepts
            key_concepts = self.extract_key_concepts(final_results)
            
            # Step 5: Update search statistics
            self.update_search_stats(legal_domain, len(final_results))
            
            # Update state with research findings
            state.update({
                'research_results': final_results,
                'enhanced_queries': enhanced_queries,
                'key_concepts': key_concepts,
                'research_metadata': {
                    'total_results_found': len(search_results),
                    'final_results_count': len(final_results),
                    'search_strategy': 'multi_stage_enhanced',
                    'domain_boost_applied': legal_domain in self.legal_domains,
                    'avg_relevance_score': np.mean([r['relevance_score'] for r in final_results]) if final_results else 0
                },
                'research_status': 'completed'
            })
            
            self.log(f"Research completed successfully with {len(final_results)} high-quality results")
            
        except Exception as e:
            self.log(f"Research error: {str(e)}", "ERROR")
            state['error'] = f"Research failed: {str(e)}"
            state['research_status'] = 'failed'
        
        return state
    
    def enhance_query(self, query: str, domain: str) -> List[str]:
        """Generate enhanced search queries with legal terminology"""
        enhanced_queries = [query]  # Start with original
        
        if domain in self.legal_domains:
            domain_info = self.legal_domains[domain]
            
            # Strategy 1: Add synonyms
            for synonym in domain_info['synonyms'][:3]:  # Use top 3 synonyms
                if synonym.lower() not in query.lower():
                    enhanced_queries.append(f"{query} {synonym}")
            
            # Strategy 2: Add key concepts
            for concept in domain_info['key_concepts'][:2]:  # Use top 2 concepts
                enhanced_queries.append(f"{concept} {query}")
            
            # Strategy 3: Add required elements
            for element in domain_info['required_elements'][:2]:
                enhanced_queries.append(f"{query} {element}")
            
            # Strategy 4: Domain-specific phrasing
            if domain == 'contract_law':
                enhanced_queries.extend([
                    f"contract {query} requirements",
                    f"legal {query} elements"
                ])
            elif domain == 'employment_law':
                enhanced_queries.extend([
                    f"employment {query} law",
                    f"workplace {query} rights"
                ])
            elif domain == 'intellectual_property':
                enhanced_queries.extend([
                    f"IP {query} protection",
                    f"intellectual property {query}"
                ])
            elif domain == 'criminal_law':
                enhanced_queries.extend([
                    f"criminal {query} law",
                    f"{query} criminal defense"
                ])
            elif domain == 'tort_law':
                enhanced_queries.extend([
                    f"tort {query} liability",
                    f"negligence {query}"
                ])
        
        # Remove duplicates and limit to 8 queries
        unique_queries = list(dict.fromkeys(enhanced_queries))[:8]
        
        self.log(f"Enhanced queries: {unique_queries}")
        return unique_queries
    
    def multi_stage_search(self, queries: List[str], top_k: int = 10) -> List[Dict]:
        """Perform multi-stage search with result aggregation"""
        all_results = []
        seen_documents = set()
        
        # Stage 1: Primary search with original query
        primary_results = self.vector_store.search(queries[0], top_k=top_k)
        for result in primary_results:
            doc_hash = hash(result['document'][:100])  # Use first 100 chars as hash
            if doc_hash not in seen_documents:
                result['search_stage'] = 'primary'
                result['query_used'] = queries[0]
                all_results.append(result)
                seen_documents.add(doc_hash)
        
        # Stage 2: Enhanced searches
        for i, query in enumerate(queries[1:], 1):
            enhanced_results = self.vector_store.search(query, top_k=max(5, top_k//2))
            for result in enhanced_results:
                doc_hash = hash(result['document'][:100])
                if doc_hash not in seen_documents:
                    result['search_stage'] = 'enhanced'
                    result['query_used'] = query
                    result['enhancement_rank'] = i
                    all_results.append(result)
                    seen_documents.add(doc_hash)
        
        # Stage 3: Semantic filtering (remove very low relevance results)
        filtered_results = [r for r in all_results if r['relevance_score'] > 0.3]
        
        self.log(f"Multi-stage search: {len(all_results)} total, {len(filtered_results)} after filtering")
        return filtered_results
    
    def rank_results(self, results: List[Dict], domain: str) -> List[Dict]:
        """Advanced ranking algorithm with multiple factors"""
        if not results:
            return results
        
        domain_info = self.legal_domains.get(domain, {})
        domain_boost = domain_info.get('search_boost', 1.0)
        
        for result in results:
            # Base score from similarity
            base_score = result['relevance_score']
            
            # Factor 1: Semantic similarity (40% weight)
            semantic_score = base_score * 0.4
            
            # Factor 2: Domain match (30% weight)
            domain_score = 0
            if result['metadata'].get('domain') == domain:
                domain_score = 0.3
            elif domain in str(result['document']).lower():
                domain_score = 0.2
            
            # Factor 3: Source authority (20% weight)
            source_score = 0
            if result['metadata'].get('type') == 'legal_principle':
                source_score = 0.2
            elif result['metadata'].get('source'):
                source_score = 0.15
            
            # Factor 4: Search stage bonus (10% weight)
            stage_score = 0
            if result.get('search_stage') == 'primary':
                stage_score = 0.1
            elif result.get('search_stage') == 'enhanced':
                stage_score = 0.05
            
            # Calculate composite score
            composite_score = semantic_score + domain_score + source_score + stage_score
            
            # Apply domain boost
            composite_score *= domain_boost
            
            # Ensure score doesn't exceed 1.0
            composite_score = min(composite_score, 1.0)
            
            result['composite_score'] = composite_score
            result['ranking_factors'] = {
                'semantic': semantic_score,
                'domain': domain_score,
                'source': source_score,
                'stage': stage_score,
                'domain_boost': domain_boost
            }
        
        # Sort by composite score
        ranked_results = sorted(results, key=lambda x: x['composite_score'], reverse=True)
        
        self.log(f"Ranking completed. Top score: {ranked_results[0]['composite_score']:.3f}")
        return ranked_results
    
    def extract_key_concepts(self, results: List[Dict]) -> List[str]:
        """Extract key legal concepts from search results"""
        if not results:
            return []
        
        concept_frequency = defaultdict(int)
        legal_terms = set()
        
        # Common legal concept patterns
        legal_patterns = [
            r'\b(?:contract|agreement|binding)\b',
            r'\b(?:negligence|liability|duty)\b',
            r'\b(?:copyright|patent|trademark)\b',
            r'\b(?:employment|termination|discrimination)\b',
            r'\b(?:criminal|defense|prosecution)\b',
            r'\b(?:statute|law|regulation)\b',
            r'\b(?:court|case|decision)\b',
            r'\b(?:rights|obligations|duties)\b'
        ]
        
        # Extract concepts from top results
        for result in results[:5]:  # Focus on top 5 results
            document = result['document'].lower()
            
            # Find legal term patterns
            for pattern in legal_patterns:
                matches = re.findall(pattern, document, re.IGNORECASE)
                for match in matches:
                    legal_terms.add(match.lower())
                    concept_frequency[match.lower()] += 1
        
        # Get most frequent concepts
        sorted_concepts = sorted(concept_frequency.items(), key=lambda x: x[1], reverse=True)
        key_concepts = [concept for concept, freq in sorted_concepts[:10] if freq > 1]
        
        self.log(f"Extracted {len(key_concepts)} key concepts")
        return key_concepts
    
    def update_search_stats(self, domain: str, result_count: int):
        """Update search statistics for monitoring"""
        self.search_stats['total_searches'] += 1
        self.search_stats['domain_distribution'][domain] += 1
        
        # Update running average
        current_avg = self.search_stats['avg_results']
        total_searches = self.search_stats['total_searches']
        self.search_stats['avg_results'] = ((current_avg * (total_searches - 1)) + result_count) / total_searches
    
    def get_search_statistics(self) -> Dict:
        """Get current search statistics"""
        return {
            'total_searches': self.search_stats['total_searches'],
            'average_results': round(self.search_stats['avg_results'], 2),
            'domain_distribution': dict(self.search_stats['domain_distribution']),
            'vector_store_stats': self.vector_store.get_collection_stats()
        }
