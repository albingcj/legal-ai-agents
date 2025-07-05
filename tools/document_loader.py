import os
import re
from typing import List, Dict
from pathlib import Path
import requests
from bs4 import BeautifulSoup

class LegalDocumentLoader:
    def __init__(self):
        self.data_dir = Path("data/legal_documents")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_sample_documents(self) -> List[Dict]:
        """Load sample legal documents for testing"""
        sample_docs = [
            {
                "title": "Contract Formation Requirements",
                "content": """
                A valid contract requires four essential elements:
                1. Offer: A clear proposal to enter into an agreement
                2. Acceptance: Unqualified agreement to the terms of the offer
                3. Consideration: Something of value exchanged between parties
                4. Mutual assent: Both parties must understand and agree to the terms
                
                Without these elements, a contract may be void or voidable.
                Courts will examine whether there was a meeting of the minds between the parties.
                """,
                "source": "Contract Law Basics",
                "type": "legal_principle",
                "domain": "contract_law"
            },
            {
                "title": "Employment At-Will Doctrine",
                "content": """
                Employment at-will is a legal doctrine that allows employers to terminate employees
                for any reason, at any time, without warning, as long as the reason is not illegal.
                
                Exceptions to at-will employment include:
                - Discrimination based on protected characteristics
                - Retaliation for whistleblowing
                - Violation of public policy
                - Breach of implied contract
                
                Most US states follow the at-will doctrine, but some have additional protections.
                """,
                "source": "Employment Law Guide",
                "type": "legal_principle",
                "domain": "employment_law"
            },
            {
                "title": "Copyright Fair Use",
                "content": """
                Fair use is a legal doctrine that permits limited use of copyrighted material
                without permission from the copyright holder. Courts consider four factors:
                
                1. Purpose and character of use (commercial vs. educational)
                2. Nature of the copyrighted work
                3. Amount and substantiality of portion used
                4. Effect on market value of original work
                
                Fair use is determined case-by-case and provides important balance between
                copyright protection and freedom of expression.
                """,
                "source": "Copyright Law Overview",
                "type": "legal_principle",
                "domain": "intellectual_property"
            },
            {
                "title": "Personal Injury Statute of Limitations",
                "content": """
                Statute of limitations sets the time limit for filing a personal injury lawsuit.
                Time limits vary by state and type of injury:
                
                - General personal injury: 1-6 years (varies by state)
                - Medical malpractice: 1-3 years from discovery
                - Product liability: 2-4 years from injury
                - Wrongful death: 1-3 years from death
                
                Discovery rule: Clock starts when injury is discovered or should have been discovered.
                Missing the deadline typically bars the claim forever.
                """,
                "source": "Personal Injury Law",
                "type": "legal_principle",
                "domain": "tort_law"
            },
            {
                "title": "Miranda Rights Requirements",
                "content": """
                Miranda rights must be read to suspects in police custody before interrogation.
                The warning must include:
                
                1. Right to remain silent
                2. Anything said can be used in court
                3. Right to an attorney
                4. Right to appointed attorney if cannot afford one
                
                Custody + Interrogation = Miranda required
                Failure to give Miranda warnings makes statements inadmissible in court.
                Spontaneous statements without questioning are still admissible.
                """,
                "source": "Criminal Procedure Law",
                "type": "legal_principle",
                "domain": "criminal_law"
            }
        ]
        
        return sample_docs
    
    def chunk_document(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split document into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def prepare_documents_for_vectorstore(self) -> List[Dict]:
        """Prepare documents for vector store ingestion"""
        documents = self.load_sample_documents()
        prepared_docs = []
        
        for doc in documents:
            # Clean and chunk the content
            chunks = self.chunk_document(doc['content'])
            
            for i, chunk in enumerate(chunks):
                prepared_docs.append({
                    'text': chunk,
                    'metadata': {
                        'title': doc['title'],
                        'source': doc['source'],
                        'type': doc['type'],
                        'domain': doc['domain'],
                        'chunk_id': i
                    }
                })
        
        return prepared_docs

# Test the document loader
if __name__ == "__main__":
    loader = LegalDocumentLoader()
    docs = loader.prepare_documents_for_vectorstore()
    
    print(f"Loaded {len(docs)} document chunks")
    for i, doc in enumerate(docs[:3]):  # Show first 3
        print(f"\n{i+1}. {doc['metadata']['title']}")
        print(f"   Domain: {doc['metadata']['domain']}")
        print(f"   Text: {doc['text'][:100]}...")