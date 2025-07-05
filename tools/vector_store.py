import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from dotenv import load_dotenv
import uuid
import logging
import threading
import time
from collections import OrderedDict

load_dotenv()
logger = logging.getLogger(__name__)

class LegalVectorStore:
    def __init__(self):
        self.db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        # Memory management
        self._embedding_cache = OrderedDict()
        self._max_cache_size = 1000
        self._cache_lock = threading.Lock()
        
        # Performance tracking
        self._stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'total_documents': 0,
            'avg_query_time': 0.0,
            'last_updated': time.time()
        }
        
        # Initialize ChromaDB
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(
                name="legal_documents",
                metadata={"description": "Legal documents and case law"}
            )
            logger.info(f"ChromaDB initialized at: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
        
        # Initialize embedding model
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _manage_cache(self):
        """Manage embedding cache size"""
        with self._cache_lock:
            if len(self._embedding_cache) > self._max_cache_size:
                # Remove oldest 20% of cache entries
                remove_count = int(self._max_cache_size * 0.2)
                for _ in range(remove_count):
                    self._embedding_cache.popitem(last=False)
                logger.debug(f"Cache cleaned: removed {remove_count} entries")
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache if available"""
        text_hash = hash(text)
        with self._cache_lock:
            if text_hash in self._embedding_cache:
                # Move to end (most recently used)
                embedding = self._embedding_cache.pop(text_hash)
                self._embedding_cache[text_hash] = embedding
                self._stats['cache_hits'] += 1
                return embedding
        return None
    
    def _cache_embedding(self, text: str, embedding: List[float]):
        """Cache embedding for future use"""
        text_hash = hash(text)
        with self._cache_lock:
            self._embedding_cache[text_hash] = embedding
        self._manage_cache()
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with caching"""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                new_embeddings = self.embedding_model.encode(uncached_texts).tolist()
                
                # Cache new embeddings and update results
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    self._cache_embedding(texts[idx], embedding)
                    embeddings[idx] = embedding
                    
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                raise
        
        return embeddings
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Add documents to the vector store with improved error handling"""
        if not documents:
            logger.warning("No documents provided to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        try:
            # Process in batches to manage memory
            batch_size = 100
            total_added = 0
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size] if metadata else None
                
                try:
                    self._process_document_batch(batch_docs, batch_metadata)
                    total_added += len(batch_docs)
                    logger.info(f"Processed batch {i//batch_size + 1}: {len(batch_docs)} documents")
                    
                except Exception as e:
                    logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                    # Continue with next batch instead of failing completely
                    continue
            
            # Update stats
            self._stats['total_documents'] = self.collection.count()
            self._stats['last_updated'] = time.time()
            
            logger.info(f"Successfully added {total_added}/{len(documents)} documents!")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def _process_document_batch(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Process a batch of documents"""
        # Generate embeddings
        embeddings = self._generate_embeddings(documents)
        
        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Prepare metadata
        if metadata is None:
            metadata = [{"source": "unknown", "type": "legal_document"} for _ in documents]
        
        # Validate metadata
        for i, meta in enumerate(metadata):
            if not isinstance(meta, dict):
                metadata[i] = {"source": "unknown", "type": "legal_document"}
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents with performance tracking"""
        start_time = time.time()
        
        try:
            logger.debug(f"Searching for: '{query}' (top {top_k} results)")
            
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])[0]
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    try:
                        formatted_results.append({
                            'document': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                            'distance': results['distances'][0][i] if results['distances'] else 1.0,
                            'relevance_score': max(0, 1 - results['distances'][0][i]) if results['distances'] else 0.0
                        })
                    except (IndexError, TypeError) as e:
                        logger.warning(f"Error processing result {i}: {e}")
                        continue
            
            # Update performance stats
            query_time = time.time() - start_time
            self._update_query_stats(query_time)
            
            logger.debug(f"Found {len(formatted_results)} relevant documents in {query_time:.3f}s")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    def _update_query_stats(self, query_time: float):
        """Update query performance statistics"""
        self._stats['total_queries'] += 1
        
        # Calculate running average
        total_queries = self._stats['total_queries']
        current_avg = self._stats['avg_query_time']
        self._stats['avg_query_time'] = ((current_avg * (total_queries - 1)) + query_time) / total_queries
    
    def get_collection_stats(self) -> Dict:
        """Get comprehensive statistics about the collection"""
        try:
            document_count = self.collection.count()
            
            # Calculate cache efficiency
            total_requests = self._stats['total_queries'] * 2  # Approximate (query + documents)
            cache_hit_rate = (self._stats['cache_hits'] / total_requests) if total_requests > 0 else 0
            
            return {
                "total_documents": document_count,
                "embedding_model": self.embedding_model_name,
                "collection_name": self.collection.name,
                "database_path": self.db_path,
                "performance_stats": {
                    "total_queries": self._stats['total_queries'],
                    "avg_query_time": f"{self._stats['avg_query_time']:.3f}s",
                    "cache_hit_rate": f"{cache_hit_rate:.1%}",
                    "cache_size": len(self._embedding_cache),
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._stats['last_updated']))
                }
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "error": str(e),
                "total_documents": 0,
                "embedding_model": self.embedding_model_name
            }
    
    def health_check(self) -> Dict[str, bool]:
        """Perform health check on the vector store"""
        health_status = {
            "database_accessible": False,
            "collection_accessible": False,
            "embedding_model_loaded": False,
            "can_perform_search": False
        }
        
        try:
            # Check database
            _ = self.client.heartbeat()
            health_status["database_accessible"] = True
            
            # Check collection
            _ = self.collection.count()
            health_status["collection_accessible"] = True
            
            # Check embedding model
            test_embedding = self.embedding_model.encode(["test"])
            health_status["embedding_model_loaded"] = len(test_embedding) > 0
            
            # Check search functionality
            if health_status["collection_accessible"] and health_status["embedding_model_loaded"]:
                test_results = self.search("test", top_k=1)
                health_status["can_perform_search"] = isinstance(test_results, list)
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        return health_status
    
    def optimize_performance(self):
        """Optimize vector store performance"""
        try:
            logger.info("Starting performance optimization...")
            
            # Clear old cache entries
            with self._cache_lock:
                if len(self._embedding_cache) > self._max_cache_size // 2:
                    old_size = len(self._embedding_cache)
                    self._embedding_cache.clear()
                    logger.info(f"Cleared cache: {old_size} entries removed")
            
            # Reset stats
            self._stats.update({
                'cache_hits': 0,
                'total_queries': 0,
                'avg_query_time': 0.0
            })
            
            logger.info("Performance optimization completed")
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            if hasattr(self, '_embedding_cache'):
                self._embedding_cache.clear()
        except:
            pass

# Test the vector store
if __name__ == "__main__":
    try:
        vs = LegalVectorStore()
        
        # Health check
        health = vs.health_check()
        print("Health Check:", health)
        
        # Test documents
        test_docs = [
            "A contract is a legally binding agreement between two or more parties.",
            "Employment at-will means an employee can be terminated for any reason.",
            "Copyright protects original works of authorship fixed in a tangible medium."
        ]
        
        test_metadata = [
            {"domain": "contract_law", "type": "definition"},
            {"domain": "employment_law", "type": "principle"},
            {"domain": "intellectual_property", "type": "definition"}
        ]
        
        # Add documents
        vs.add_documents(test_docs, test_metadata)
        
        # Test search
        results = vs.search("What is a contract?", top_k=2)
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Score: {result['relevance_score']:.3f}")
            print(f"Text: {result['document'][:100]}...")
            print(f"Metadata: {result['metadata']}")
        
        # Show stats
        stats = vs.get_collection_stats()
        print(f"\nCollection Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Test failed: {e}")
