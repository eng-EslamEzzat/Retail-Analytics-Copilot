"""Document retrieval using TF-IDF."""
from typing import List, Dict, Any, Tuple
from pathlib import Path
import re
from collections import Counter
import math


class Chunk:
    """Represents a document chunk."""
    def __init__(self, id: str, content: str, source: str, score: float = 0.0):
        self.id = id
        self.content = content
        self.source = source
        self.score = score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "score": self.score
        }


class TFIDFRetriever:
    """Simple TF-IDF based document retriever."""
    
    def __init__(self, docs_dir: str = "docs"):
        """Initialize retriever with documents directory."""
        self.docs_dir = Path(docs_dir)
        self.chunks: List[Chunk] = []
        self.vocab: Dict[str, float] = {}  # IDF scores
        self.chunk_vectors: List[Dict[str, float]] = []  # TF vectors per chunk
        
        self._load_documents()
        self._build_index()
    
    def _load_documents(self):
        """Load and chunk documents from docs directory."""
        chunk_id = 0
        
        for doc_file in self.docs_dir.glob("*.md"):
            content = doc_file.read_text(encoding='utf-8')
            source = doc_file.stem
            
            # Simple paragraph-level chunking
            paragraphs = re.split(r'\n\s*\n', content)
            
            for para in paragraphs:
                para = para.strip()
                if len(para) > 20:  # Skip very short paragraphs
                    chunk = Chunk(
                        id=f"{source}::chunk{chunk_id}",
                        content=para,
                        source=source
                    )
                    self.chunks.append(chunk)
                    chunk_id += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (lowercase, alphanumeric)."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _build_index(self):
        """Build TF-IDF index."""
        # Count document frequency for each term
        doc_freq: Dict[str, int] = Counter()
        all_chunk_tokens: List[List[str]] = []
        
        for chunk in self.chunks:
            tokens = self._tokenize(chunk.content)
            all_chunk_tokens.append(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
        
        # Calculate IDF
        num_docs = len(self.chunks)
        for term, df in doc_freq.items():
            self.vocab[term] = math.log(num_docs / (df + 1))
        
        # Calculate TF for each chunk
        for tokens in all_chunk_tokens:
            tf: Dict[str, float] = Counter(tokens)
            total_terms = len(tokens)
            tf_vector = {term: count / total_terms for term, count in tf.items()}
            self.chunk_vectors.append(tf_vector)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """
        Retrieve top-k chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of chunks to return
            
        Returns:
            List of Chunk objects sorted by relevance score
        """
        query_tokens = self._tokenize(query)
        query_tf: Dict[str, float] = Counter(query_tokens)
        total_query_terms = len(query_tokens)
        if total_query_terms == 0:
            return []
        
        query_vector = {term: count / total_query_terms for term, count in query_tf.items()}
        
        # Calculate cosine similarity
        scores = []
        for i, chunk_vector in enumerate(self.chunk_vectors):
            score = 0.0
            query_norm = 0.0
            chunk_norm = 0.0
            
            # Dot product and norms
            all_terms = set(query_vector.keys()) | set(chunk_vector.keys())
            for term in all_terms:
                query_tf_idf = query_vector.get(term, 0) * self.vocab.get(term, 0)
                chunk_tf_idf = chunk_vector.get(term, 0) * self.vocab.get(term, 0)
                
                score += query_tf_idf * chunk_tf_idf
                query_norm += query_tf_idf ** 2
                chunk_norm += chunk_tf_idf ** 2
            
            if query_norm > 0 and chunk_norm > 0:
                score = score / (math.sqrt(query_norm) * math.sqrt(chunk_norm))
            
            scores.append((score, i))
        
        # Sort by score and return top-k
        scores.sort(reverse=True, key=lambda x: x[0])
        results = []
        for score, idx in scores[:top_k]:
            chunk = self.chunks[idx]
            chunk.score = score
            results.append(chunk)
        
        return results

