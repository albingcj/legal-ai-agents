from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from tools.llm_client import LMStudioClient
import re
from datetime import datetime
import json

class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Legal Analysis Agent",
            description="Generate comprehensive legal analysis with proper citations and disclaimers"
        )
        
        # Initialize LLM client
        self.llm_client = LMStudioClient()
        
        # Legal prompt templates for different domains
        self.prompt_templates = {
            'contract_law': {
                'system': """You are a legal research assistant specializing in contract law. 
                Provide comprehensive analysis based on legal principles, but always include appropriate disclaimers.
                Focus on contract formation, performance, and enforcement issues.""",
                'structure': [
                    "Legal Principles",
                    "Key Requirements/Elements", 
                    "Potential Issues or Exceptions",
                    "Practical Considerations"
                ]
            },
            'employment_law': {
                'system': """You are a legal research assistant specializing in employment law.
                Analyze workplace legal issues while emphasizing both employer and employee perspectives.
                Focus on rights, obligations, and compliance requirements.""",
                'structure': [
                    "Legal Framework",
                    "Rights and Obligations",
                    "Compliance Requirements",
                    "Potential Risks"
                ]
            },
            'intellectual_property': {
                'system': """You are a legal research assistant specializing in intellectual property law.
                Analyze IP protection, infringement, and enforcement issues with attention to fair use and licensing.
                Focus on creation, protection, and enforcement of IP rights.""",
                'structure': [
                    "IP Protection Scope",
                    "Legal Requirements",
                    "Infringement Analysis",
                    "Fair Use Considerations"
                ]
            },
            'criminal_law': {
                'system': """You are a legal research assistant specializing in criminal law.
                Analyze criminal procedure and defense issues with attention to constitutional rights.
                Focus on elements, defenses, and procedural requirements.""",
                'structure': [
                    "Legal Elements",
                    "Constitutional Considerations",
                    "Available Defenses",
                    "Procedural Requirements"
                ]
            },
            'tort_law': {
                'system': """You are a legal research assistant specializing in tort law.
                Analyze liability, negligence, and damages issues with focus on duty and causation.
                Focus on duty, breach, causation, and damages.""",
                'structure': [
                    "Duty and Standard of Care",
                    "Breach Analysis",
                    "Causation Requirements",
                    "Damages Assessment"
                ]
            },
            'general': {
                'system': """You are a legal research assistant providing general legal information.
                Analyze legal issues comprehensively while emphasizing the need for professional legal counsel.
                Focus on general legal principles and considerations.""",
                'structure': [
                    "Legal Overview",
                    "Key Considerations",
                    "Potential Issues",
                    "Next Steps"
                ]
            }
        }
        
        # Standard legal disclaimer
        self.legal_disclaimer = """
        **IMPORTANT LEGAL DISCLAIMER:**
        This information is provided for educational purposes only and does not constitute legal advice. 
        The law varies by jurisdiction and individual circumstances. For specific legal matters, 
        please consult with a qualified attorney in your jurisdiction.
        """
        
        # Citation patterns for matching
        self.citation_patterns = [
            r'\b(?:contract|agreement|consideration|offer|acceptance)\b',
            r'\b(?:employment|termination|discrimination|at-will)\b',
            r'\b(?:copyright|patent|trademark|fair use)\b',
            r'\b(?:negligence|liability|duty|damages)\b',
            r'\b(?:criminal|miranda|rights|defense)\b'
        ]
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main analysis pipeline"""
        self.log("Starting legal analysis process")
        
        # Validate required state
        if not self.validate_state(state, ['research_results', 'legal_domain', 'query']):
            state['error'] = "Missing required state for analysis"
            return state
        
        try:
            research_results = state['research_results']
            legal_domain = state['legal_domain']
            original_query = state['query']
            
            self.log(f"Analyzing {len(research_results)} research results for domain: {legal_domain}")
            
            # Step 1: Build comprehensive legal context
            legal_context = self.build_legal_context(research_results, original_query)
            self.log(f"Built context with {len(legal_context.split())} words")
            
            # Step 2: Generate legal analysis
            analysis = self.generate_legal_analysis(legal_context, original_query, legal_domain)
            self.log("Generated legal analysis")
            
            # Step 3: Add citations
            analysis_with_citations = self.add_citations(analysis, research_results)
            self.log("Added citations to analysis")
            
            # Step 4: Format final response
            formatted_response = self.format_legal_response(
                analysis_with_citations, research_results, legal_domain
            )
            self.log("Formatted final response")
            
            # Step 5: Calculate confidence score
            confidence_score = self.calculate_confidence_score(research_results, analysis)
            
            # Update state with analysis
            state.update({
                'legal_analysis': analysis,
                'analysis_with_citations': analysis_with_citations,
                'final_response': formatted_response,
                'confidence_score': confidence_score,
                'analysis_metadata': {
                    'context_length': len(legal_context.split()),
                    'analysis_length': len(analysis.split()),
                    'citation_count': len(self.extract_citations(analysis_with_citations)),
                    'domain_template_used': legal_domain,
                    'processing_timestamp': datetime.now().isoformat()
                },
                'analysis_status': 'completed'
            })
            
            self.log(f"Analysis completed successfully with confidence: {confidence_score:.2f}")
            
        except Exception as e:
            self.log(f"Analysis error: {str(e)}", "ERROR")
            state['error'] = f"Analysis failed: {str(e)}"
            state['analysis_status'] = 'failed'
        
        return state
    
    def process_stream(self, state: Dict[str, Any]):
        """Main analysis pipeline with streaming support"""
        self.log("Starting legal analysis process with streaming")
        
        # Validate required state
        if not self.validate_state(state, ['research_results', 'legal_domain', 'query']):
            state['error'] = "Missing required state for analysis"
            yield state
            return
        
        try:
            research_results = state['research_results']
            legal_domain = state['legal_domain']
            original_query = state['query']
            
            self.log(f"Analyzing {len(research_results)} research results for domain: {legal_domain}")
            
            # Step 1: Build comprehensive legal context
            legal_context = self.build_legal_context(research_results, original_query)
            self.log(f"Built context with {len(legal_context.split())} words")
            
            # Step 2: Generate legal analysis with streaming
            full_analysis = ""
            for chunk in self.generate_legal_analysis_stream(legal_context, original_query, legal_domain):
                full_analysis += chunk
                yield {
                    'type': 'analysis_chunk',
                    'content': chunk,
                    'full_content': full_analysis
                }
            
            self.log("Generated legal analysis with streaming")
            
            # Step 3: Add citations
            analysis_with_citations = self.add_citations(full_analysis, research_results)
            self.log("Added citations to analysis")
            
            # Step 4: Format final response
            formatted_response = self.format_legal_response(
                analysis_with_citations, research_results, legal_domain
            )
            self.log("Formatted final response")
            
            # Step 5: Calculate confidence score
            confidence_score = self.calculate_confidence_score(research_results, full_analysis)
            
            # Update state with analysis
            state.update({
                'legal_analysis': full_analysis,
                'analysis_with_citations': analysis_with_citations,
                'final_response': formatted_response,
                'confidence_score': confidence_score,
                'analysis_metadata': {
                    'context_length': len(legal_context.split()),
                    'analysis_length': len(full_analysis.split()),
                    'num_sources': len(research_results),
                    'domain': legal_domain
                },
                'analysis_status': 'completed'
            })
            
            # Final yield with complete state
            yield {
                'type': 'analysis_complete',
                'state': state
            }
            
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            self.log(error_msg, level="ERROR")
            state['error'] = error_msg
            state['analysis_status'] = 'failed'
            yield {
                'type': 'error',
                'message': error_msg,
                'state': state
            }

    def build_legal_context(self, research_results: List[Dict], query: str) -> str:
        """Build comprehensive legal context from research results"""
        if not research_results:
            return ""
        
        context_parts = []
        
        # Add query context
        context_parts.append(f"Query: {query}")
        context_parts.append("=" * 50)
        
        # Group results by relevance tiers
        high_relevance = [r for r in research_results if r.get('composite_score', 0) > 0.7]
        medium_relevance = [r for r in research_results if 0.4 <= r.get('composite_score', 0) <= 0.7]
        
        # Add high relevance results (most important)
        if high_relevance:
            context_parts.append("\n**PRIMARY LEGAL PRINCIPLES:**")
            for i, result in enumerate(high_relevance[:3], 1):
                source_info = f"[Source: {result['metadata'].get('title', 'Legal Document')}]"
                context_parts.append(f"{i}. {source_info}")
                context_parts.append(result['document'])
                context_parts.append("")
        
        # Add medium relevance results (supporting information)
        if medium_relevance:
            context_parts.append("\n**SUPPORTING INFORMATION:**")
            for i, result in enumerate(medium_relevance[:3], 1):
                source_info = f"[Source: {result['metadata'].get('title', 'Legal Document')}]"
                context_parts.append(f"{i}. {source_info}")
                context_parts.append(result['document'])
                context_parts.append("")
        
        # Add domain-specific context
        domain_context = self.get_domain_context(research_results)
        if domain_context:
            context_parts.append("\n**LEGAL DOMAIN CONTEXT:**")
            context_parts.append(domain_context)
        
        return "\n".join(context_parts)
    
    def get_domain_context(self, research_results: List[Dict]) -> str:
        """Get domain-specific context information"""
        domains = [r['metadata'].get('domain') for r in research_results if r['metadata'].get('domain')]
        if not domains:
            return ""
        
        domain_counts = {}
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        primary_domain = max(domain_counts, key=domain_counts.get)
        
        domain_info = {
            'contract_law': "This involves contract formation, performance, and breach issues.",
            'employment_law': "This involves workplace rights, obligations, and employment relationships.",
            'intellectual_property': "This involves creation, protection, and enforcement of IP rights.",
            'criminal_law': "This involves criminal procedure, rights, and defense strategies.",
            'tort_law': "This involves civil liability, negligence, and damages."
        }
        
        return domain_info.get(primary_domain, "This involves general legal principles and considerations.")
    
    def generate_legal_analysis(self, context: str, query: str, domain: str) -> str:
        """Generate legal analysis using LLM"""
        # Get domain-specific template
        template = self.prompt_templates.get(domain, self.prompt_templates['general'])
        
        # Build the prompt
        system_prompt = template['system']
        structure = template['structure']
        
        user_prompt = f"""
        Based on the following legal context, provide a comprehensive analysis of this question: "{query}"

        {context}

        Please structure your response with the following sections:
        {chr(10).join([f"- {section}" for section in structure])}

        Requirements:
        1. Base your analysis strictly on the provided legal context
        2. Be precise and analytical in your explanations
        3. Highlight key legal principles and their applications
        4. Identify potential issues or exceptions
        5. Use clear, professional legal language
        6. Do not provide specific legal advice - focus on legal information and analysis
        7. Reference the sources when discussing specific legal principles

        Provide a thorough but concise analysis.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate response with streaming support
        if hasattr(self, '_use_streaming') and self._use_streaming:
            # Use streaming if enabled
            response = ""
            for chunk in self.generate_response_stream(messages, temperature=0.1):
                response += chunk
        else:
            # Use regular response
            response = self.llm_client.generate_response(messages, temperature=0.1)
        
        return response
    
    def generate_legal_analysis_stream(self, context: str, query: str, domain: str):
        """Generate legal analysis using LLM with streaming"""
        # Get domain-specific template
        template = self.prompt_templates.get(domain, self.prompt_templates['general'])
        
        # Build the prompt
        system_prompt = template['system']
        structure = template['structure']
        
        user_prompt = f"""
        Based on the following legal context, provide a comprehensive analysis of this question: "{query}"

        {context}

        Please structure your response with the following sections:
        {chr(10).join([f"- {section}" for section in structure])}

        Requirements:
        1. Base your analysis strictly on the provided legal context
        2. Be precise and analytical in your explanations
        3. Highlight key legal principles and their applications
        4. Identify potential issues or exceptions
        5. Use clear, professional legal language
        6. Do not provide specific legal advice - focus on legal information and analysis
        7. Reference the sources when discussing specific legal principles

        Provide a thorough but concise analysis.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate streaming response
        for chunk in self.llm_client.generate_response_stream(messages, temperature=0.1):
            yield chunk

    def add_citations(self, analysis: str, sources: List[Dict]) -> str:
        """Add proper citations to the analysis"""
        if not sources:
            return analysis
        
        # Create citation mapping
        citations = {}
        citation_list = []
        
        for i, source in enumerate(sources[:6], 1):  # Limit to top 6 sources
            title = source['metadata'].get('title', f'Legal Document {i}')
            domain = source['metadata'].get('domain', 'General Law')
            score = source.get('composite_score', 0)
            
            citation_text = f"[{i}] {title} - {domain.replace('_', ' ').title()} (Relevance: {score:.2f})"
            citations[i] = citation_text
            citation_list.append(citation_text)
        
        # Add citation markers in text (simple approach)
        # Look for key concepts and add citations
        analysis_with_citations = analysis
        
        # Add citations at the end
        if citation_list:
            analysis_with_citations += "\n\n**Sources:**\n"
            analysis_with_citations += "\n".join(citation_list)
        
        return analysis_with_citations
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract citation references from text"""
        citation_pattern = r'\[(\d+)\][^[]*?(?=\[|\n|$)'
        matches = re.findall(citation_pattern, text)
        return matches
    
    def format_legal_response(self, analysis: str, sources: List[Dict], domain: str) -> Dict[str, Any]:
        """Format the final legal response"""
        # Extract key points from analysis
        key_points = self.extract_key_points(analysis)
        
        # Create source references
        source_references = []
        for source in sources[:5]:  # Top 5 sources
            source_references.append({
                'title': source['metadata'].get('title', 'Legal Document'),
                'domain': source['metadata'].get('domain', 'general'),
                'relevance_score': source.get('composite_score', 0),
                'source': source['metadata'].get('source', 'Legal Database'),
                'preview': source['document'][:200] + "..." if len(source['document']) > 200 else source['document']
            })
        
        formatted_response = {
            'analysis': analysis,
            'key_points': key_points,
            'legal_domain': domain.replace('_', ' ').title(),
            'disclaimer': self.legal_disclaimer.strip(),
            'sources': source_references,
            'citation_count': len(self.extract_citations(analysis)),
            'analysis_timestamp': datetime.now().isoformat(),
            'response_type': 'legal_analysis'
        }
        
        return formatted_response
    
    def extract_key_points(self, analysis: str) -> List[str]:
        """Extract key points from the analysis"""
        key_points = []
        
        # Look for numbered or bulleted lists
        lines = analysis.split('\n')
        for line in lines:
            line = line.strip()
            # Check for various list patterns
            if (line.startswith(('1.', '2.', '3.', '4.', '5.')) or 
                line.startswith(('•', '-', '*')) or
                line.startswith(('First', 'Second', 'Third', 'Finally'))):
                # Clean up the line
                cleaned_line = re.sub(r'^[\d\.\-\*\•\s]+', '', line).strip()
                if len(cleaned_line) > 20:  # Only substantial points
                    key_points.append(cleaned_line)
        
        # If no structured points found, extract sentences with key legal terms
        if not key_points:
            sentences = re.split(r'[.!?]+', analysis)
            for sentence in sentences:
                sentence = sentence.strip()
                # Look for sentences with legal keywords
                if any(keyword in sentence.lower() for keyword in 
                       ['must', 'required', 'shall', 'liability', 'rights', 'obligations', 'law', 'legal']):
                    if 30 <= len(sentence) <= 150:  # Reasonable length
                        key_points.append(sentence + '.')
        
        return key_points[:6]  # Limit to 6 key points
    
    def calculate_confidence_score(self, research_results: List[Dict], analysis: str) -> float:
        """Calculate confidence score for the analysis"""
        if not research_results:
            return 0.0
        
        # Factor 1: Average relevance of research results (40%)
        avg_relevance = sum(r.get('composite_score', 0) for r in research_results) / len(research_results)
        relevance_factor = avg_relevance * 0.4
        
        # Factor 2: Number of high-quality sources (30%)
        high_quality_sources = len([r for r in research_results if r.get('composite_score', 0) > 0.6])
        source_factor = min(high_quality_sources / 5, 1.0) * 0.3  # Normalize to max 5 sources
        
        # Factor 3: Analysis completeness (20%)
        analysis_length = len(analysis.split())
        completeness_factor = min(analysis_length / 300, 1.0) * 0.2  # Normalize to 300 words
        
        # Factor 4: Domain match (10%)
        domain_matches = len([r for r in research_results if r['metadata'].get('domain')])
        domain_factor = min(domain_matches / len(research_results), 1.0) * 0.1
        
        confidence_score = relevance_factor + source_factor + completeness_factor + domain_factor
        
        return min(confidence_score, 1.0)  # Cap at 1.0
