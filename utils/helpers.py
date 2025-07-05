import re
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep legal punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
    
    return text.strip()

def extract_legal_terms(text: str) -> List[str]:
    """Extract legal terms from text"""
    legal_patterns = [
        r'\b(?:contract|agreement|consideration|breach|offer|acceptance)\b',
        r'\b(?:negligence|liability|duty|damages|tort|injury)\b',
        r'\b(?:copyright|patent|trademark|intellectual property|fair use)\b',
        r'\b(?:employment|termination|discrimination|at-will|harassment)\b',
        r'\b(?:criminal|miranda|rights|defense|prosecution|arrest)\b',
        r'\b(?:statute|law|regulation|court|case|decision|ruling)\b'
    ]
    
    terms = set()
    for pattern in legal_patterns:
        matches = re.findall(pattern, text.lower(), re.IGNORECASE)
        terms.update(matches)
    
    return list(terms)

def format_legal_citation(title: str, source: str, relevance: float) -> str:
    """Format legal citation"""
    return f"{title} - {source} (Relevance: {relevance:.2f})"

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def generate_query_hash(query: str) -> str:
    """Generate hash for query caching"""
    return hashlib.md5(query.encode()).hexdigest()[:8]

def validate_legal_query(query: str) -> Dict[str, Any]:
    """Validate and analyze legal query"""
    if not query or len(query.strip()) < 5:
        return {
            'valid': False,
            'error': 'Query too short',
            'suggestions': ['Please provide a more detailed legal question']
        }
    
    if len(query) > 1000:
        return {
            'valid': False,
            'error': 'Query too long',
            'suggestions': ['Please shorten your question to focus on specific legal issues']
        }
    
    # Check for legal context
    legal_terms = extract_legal_terms(query)
    
    return {
        'valid': True,
        'legal_terms_found': legal_terms,
        'has_legal_context': len(legal_terms) > 0,
        'query_length': len(query.split()),
        'complexity': 'simple' if len(query.split()) < 10 else 'complex'
    }

def format_processing_time(seconds: float) -> str:
    """Format processing time for display"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def create_error_response(error_message: str, error_type: str = "general") -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        'analysis': f"""
        I apologize, but I encountered an error while processing your request.
        
        **Error Type:** {error_type.title()}
        **Details:** {error_message}
        
        **What you can try:**
        1. Rephrase your question more specifically
        2. Focus on a single legal issue
        3. Check if your question relates to supported legal areas
        4. Try again in a few moments
        
        For specific legal advice, please consult with a qualified attorney.
        """,
        'key_points': [
            "An error occurred during processing",
            "Try rephrasing your question",
            "Consult an attorney for specific legal advice"
        ],
        'legal_domain': 'Error Response',
        'disclaimer': """
        **IMPORTANT:** This is an error response. For legal matters, 
        always consult with a qualified attorney in your jurisdiction.
        """,
        'sources': [],
        'citation_count': 0,
        'response_type': 'error_response',
        'error_details': {
            'type': error_type,
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        }
    }

def log_workflow_event(event: str, details: Dict[str, Any], level: str = "INFO"):
    """Log workflow events"""
    logger = logging.getLogger("workflow")
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'event': event,
        'details': details
    }
    
    if level == "ERROR":
        logger.error(json.dumps(log_entry))
    elif level == "WARNING":
        logger.warning(json.dumps(log_entry))
    else:
        logger.info(json.dumps(log_entry))

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for exports"""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    
    return filename

def parse_legal_domain_from_query(query: str) -> Optional[str]:
    """Parse legal domain from query text"""
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
    
    return None

class LegalTextProcessor:
    """Advanced legal text processing utilities"""
    
    @staticmethod
    def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from legal text"""
        # Simple noun phrase extraction
        phrases = []
        
        # Look for legal noun phrases
        legal_phrase_patterns = [
            r'\b(?:legal\s+\w+|court\s+\w+|law\s+\w+)\b',
            r'\b(?:\w+\s+law|\w+\s+rights|\w+\s+liability)\b',
            r'\b(?:statute\s+of\s+\w+|burden\s+of\s+\w+)\b'
        ]
        
        for pattern in legal_phrase_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            phrases.extend(matches)
        
        # Remove duplicates and limit
        unique_phrases = list(dict.fromkeys(phrases))
        return unique_phrases[:max_phrases]
    
    @staticmethod
    def identify_legal_entities(text: str) -> Dict[str, List[str]]:
        """Identify legal entities in text"""
        entities = {
            'courts': [],
            'statutes': [],
            'cases': [],
            'legal_concepts': []
        }
        
        # Court patterns
        court_patterns = [
            r'\b(?:Supreme\s+Court|Court\s+of\s+Appeals|District\s+Court)\b',
            r'\b(?:\w+\s+v\.\s+\w+)\b'  # Case names
        ]
        
        for pattern in court_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['courts'].extend(matches)
        
        # Legal concept patterns
        concept_patterns = [
            r'\b(?:due\s+process|equal\s+protection|reasonable\s+doubt)\b',
            r'\b(?:miranda\s+rights|search\s+and\s+seizure)\b'
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['legal_concepts'].extend(matches)
        
        return entities
