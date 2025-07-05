import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from dotenv import load_dotenv
import uuid

load_dotenv()

class LegalVectorStore:
    def __init__(self):
        self.db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name="legal_documents",
            metadata={"description": "Legal documents and case law"}
        )
        
        # Initialize embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        print("Embedding model loaded successfully!")
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Add documents to the vector store"""
        if not documents:
            return
            
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Prepare metadata
        if metadata is None:
            metadata = [{"source": "unknown", "type": "legal_document"} for _ in documents]
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )
        
        print(f"Successfully added {len(documents)} documents!")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        print(f"Searching for: '{query}' (top {top_k} results)")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'relevance_score': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        print(f"Found {len(formatted_results)} relevant documents")
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "total_documents": count,
            "embedding_model": self.embedding_model_name,
            "collection_name": self.collection.name
        }

# Test the vector store
if __name__ == "__main__":
    vs = LegalVectorStore()
    
    # Test documents
    test_docs = [
        "A contract is a legally binding agreement between two or more parties.",
        "Employment at-will means an employee can be terminated for any reason.",
        "Copyright protects original works of authorship fixed in a tangible medium."
    ]
    
    vs.add_documents(test_docs)
    
    # Test search
    results = vs.search("What is a contract?")
    for result in results:
        print(f"Score: {result['relevance_score']:.3f} - {result['document'][:100]}...")