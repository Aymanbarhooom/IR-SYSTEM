# models/bm25_model.py
import json
import hashlib
from rank_bm25 import BM25Okapi
from config import PROCESSED_DOCUMENTS_FILE
from functools import lru_cache


class BM25Model:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern: ensure only one instance exists"""
        if cls._instance is None:
            cls._instance = super(BM25Model, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize only once"""
        if not BM25Model._initialized:
            self.doc_ids = []
            self.documents = []   # tokens
            self.bm25 = None
            self.index_hash = None
            BM25Model._initialized = True
            self.build_index()
    
    def load_documents(self):
        """Load documents only if not already loaded"""
        if self.documents:
            return
            
        print("📄 Loading processed documents for BM25...")
        with open(PROCESSED_DOCUMENTS_FILE, "r", encoding="utf-8") as f:
            docs = json.load(f)
        
        for doc_id, doc_data in docs.items():
            self.doc_ids.append(doc_id)
            text = doc_data.get("processed_text", "")
            self.documents.append(text.split())
        
        print(f"✅ Loaded {len(self.documents):,} documents")
    
    def build_index(self):
        """Build index only if not already built"""
        # Check if index already exists
        if self.bm25 is not None and self.documents:
            print(f"✅ BM25 Index already built with {len(self.doc_ids):,} documents. Skipping rebuild.")
            return
        
        print("🔨 Building BM25 Model...")
        self.load_documents()
        
        if not self.documents:
            raise ValueError("No documents loaded!")
        
        print(f"Building BM25 on {len(self.documents):,} documents...")
        self.bm25 = BM25Okapi(self.documents)
        
        self.index_hash = hashlib.md5(
            str(len(self.documents)).encode() + 
            str(self.doc_ids[-100:]).encode()
        ).hexdigest()
        
        print(f"✅ BM25 Index ready with {len(self.doc_ids):,} documents.")
    
    @lru_cache(maxsize=128)
    def _cached_search(self, query_tuple, top_k):
        """Internal cached search method"""
        query_tokens = " ".join(query_tuple).split()
        scores = self.bm25.get_scores(query_tokens)
        
        ranked = sorted(
            zip(self.doc_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:top_k]
    
    def search(self, query, top_k=10):
        """Search with caching for repeated queries"""
        # Use cache for repeated queries
        query_tuple = tuple(query.split())
        return self._cached_search(query_tuple, top_k)
    
    def refresh_index(self):
        """Force rebuild index (useful if documents changed)"""
        print("🔄 Refreshing BM25 index...")
        BM25Model._initialized = False
        self.doc_ids = []
        self.documents = []
        self.bm25 = None
        self.__init__()
        print("✅ BM25 index refreshed")