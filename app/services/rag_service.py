import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re
import time
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHROMA_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_db")
POLICIES_DIR = os.path.join(PROJECT_ROOT, "data", "hr_policies")
COLLECTION_NAME = "hr_policies"

class RAGService:
    def __init__(self):
        """Initialize RAG service with models and vector database."""
        init_start = time.time()
        
        # Load embedding model
        print("Loading sentence transformer models...")
        embedding_start = time.time()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_time = time.time() - embedding_start
        print(f"  ✓ Embedding model loaded in {embedding_time:.2f}s")
        
        # Load reranker model
        reranker_start = time.time()
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker_time = time.time() - reranker_start
        print(f"  ✓ Reranker model loaded in {reranker_time:.2f}s")
        
        # Connect to ChromaDB
        print(f"Connecting to ChromaDB at {CHROMA_DIR}...")
        chroma_start = time.time()
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        if self.collection.count() == 0:
            self._index_policy_pdfs()
        chroma_time = time.time() - chroma_start
        print(f"  ✓ ChromaDB connected in {chroma_time:.2f}s")
        
        # Load documents for BM25
        bm25_start = time.time()
        self._load_documents_for_bm25()
        bm25_time = time.time() - bm25_start
        print(f"  ✓ BM25 documents loaded in {bm25_time:.2f}s")
        
        # Initialize OpenRouter client for LLM generation
        client_start = time.time()
        load_dotenv()
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        # Use specific free model instead of router for more predictable response format
        self.llm_model = "inclusionai/ling-3.0-flash-fin:free"
        
        if self.openrouter_api_key:
            self.openai_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key
            )
            print("OpenRouter client initialized for LLM generation")
        else:
            self.openai_client = None
            print("OPENROUTER_API_KEY not set - will use template-based answers")
        client_time = time.time() - client_start
        
        total_init_time = time.time() - init_start
        print(f"\n✓ RAG service initialization complete in {total_init_time:.2f}s")
        print(f"  Breakdown: embedding={embedding_time:.2f}s, reranker={reranker_time:.2f}s, chroma={chroma_time:.2f}s, bm25={bm25_time:.2f}s, client={client_time:.2f}s")
        
    def _load_documents_for_bm25(self):
        """Load documents from ChromaDB for BM25 indexing."""
        print("Loading documents for BM25 indexing...")
        result = self.collection.get()
        
        self.documents = result['documents']
        self.metadatas = result['metadatas']
        self.ids = result['ids']
        
        # Tokenize documents for BM25
        self.tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        print(f"Loaded {len(self.documents)} documents for BM25")

    def _index_policy_pdfs(self):
        """Build the policy collection on a fresh deployment if it is absent/empty."""
        documents = []
        metadatas = []
        ids = []
        for filename in sorted(os.listdir(POLICIES_DIR)) if os.path.isdir(POLICIES_DIR) else []:
            if not filename.lower().endswith(".pdf"):
                continue
            path = os.path.join(POLICIES_DIR, filename)
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            if text:
                documents.append(text)
                metadatas.append({"source": filename})
                ids.append(os.path.splitext(filename)[0])

        if not documents:
            raise RuntimeError(f"No readable HR policy PDFs found in {POLICIES_DIR}")

        embeddings = self.embedding_model.encode(documents).tolist()
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        print(f"Indexed {len(documents)} HR policy documents into ChromaDB")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Convert to lowercase and split on whitespace/punctuation
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform vector search using ChromaDB."""
        query_embedding = self.embedding_model.encode([query])
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        vector_results = []
        for i in range(len(results['documents'][0])):
            vector_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'id': results['ids'][0][i],
                'score': results['distances'][0][i]  # Actually cosine distance
            })
        
        return vector_results
    
    def _bm25_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform BM25 search."""
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k documents
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        bm25_results = []
        for idx in top_indices:
            bm25_results.append({
                'document': self.documents[idx],
                'metadata': self.metadatas[idx],
                'id': self.ids[idx],
                'score': doc_scores[idx]
            })
        
        return bm25_results
    
    def _hybrid_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid search combining BM25 and vector search."""
        # Get results from both methods
        vector_results = self._vector_search(query, top_k=top_k)
        bm25_results = self._bm25_search(query, top_k=top_k)
        
        # Combine and deduplicate by document ID
        combined = {}
        
        # Add vector results (use inverse distance as score)
        for result in vector_results:
            doc_id = result['id']
            combined[doc_id] = {
                'document': result['document'],
                'metadata': result['metadata'],
                'id': doc_id,
                'vector_score': 1 - result['score'],  # Convert distance to similarity
                'bm25_score': 0
            }
        
        # Add BM25 results and update scores
        for result in bm25_results:
            doc_id = result['id']
            if doc_id in combined:
                combined[doc_id]['bm25_score'] = result['score']
            else:
                combined[doc_id] = {
                    'document': result['document'],
                    'metadata': result['metadata'],
                    'id': doc_id,
                    'vector_score': 0,
                    'bm25_score': result['score']
                }
        
        # Calculate hybrid score (weighted average)
        for doc_id, result in combined.items():
            # Normalize scores to 0-1 range (simple approach)
            vector_score = result['vector_score']
            bm25_score = result['bm25_score'] / max(score['bm25_score'] for score in combined.values()) if any(s['bm25_score'] > 0 for s in combined.values()) else 0
            
            # Hybrid score: 60% vector, 40% BM25
            result['hybrid_score'] = 0.6 * vector_score + 0.4 * bm25_score
        
        # Sort by hybrid score and return top-k
        results = sorted(combined.values(), key=lambda x: x['hybrid_score'], reverse=True)[:top_k]
        
        return results
    
    def _rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[int]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get cross-encoder scores
        scores = self.reranker.predict(pairs)
        
        # Sort by scores and return top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        return top_indices
    
    def _generate_answer(self, query: str, context_docs: List[str]) -> str:
        """Generate answer using retrieved context with LLM or fallback to template."""
        if not self.openai_client:
            # Fallback to template-based answer
            context = "\n\n".join([f"Excerpt: {doc[:200]}..." for doc in context_docs])
            return f"""Based on the HR policies, here's what I found regarding "{query}":

{context}

Note: OPENROUTER_API_KEY is not set or LLM call failed. Using template-based answer. Add your OpenRouter API key to .env file for LLM-generated answers."""
        
        try:
            # Prepare context for LLM
            context_text = "\n\n".join([f"Document excerpt: {doc}" for doc in context_docs])
            
            # System prompt for grounded answers
            system_prompt = """You are an HR policy assistant. Answer the user's question based ONLY on the provided context excerpts from HR policy documents. 

Rules:
1. Use only the information provided in the context excerpts
2. If the context doesn't contain information to answer the question, say so plainly - do not invent information
3. Be concise and direct in your answers
4. If the context contains relevant information, provide a clear, specific answer
5. Do not include any information not found in the context"""

            user_prompt = f"""Question: {query}

Context excerpts from HR policies:
{context_text}

Answer the question based only on the provided context:"""

            # Call OpenRouter API
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            # Fallback to template-based answer on any error
            print(f"LLM generation failed: {e}")
            context = "\n\n".join([f"Excerpt: {doc[:200]}..." for doc in context_docs])
            return f"""Based on the HR policies, here's what I found regarding "{query}":

{context}

Note: LLM generation failed ({str(e)}). Using template-based answer. Check your OPENROUTER_API_KEY and internet connection."""
    
    def ask(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Main RAG pipeline: hybrid search + rerank + answer generation.
        
        Args:
            query: User question
            top_k: Number of final results to return
            
        Returns:
            Dictionary with answer, retrieved documents, and timing breakdown
        """
        pipeline_start = time.time()
        timing = {}
        
        # Step 1: Hybrid search
        print(f"\n--- RAG Query: '{query}' ---")
        hybrid_start = time.time()
        hybrid_results = self._hybrid_search(query, top_k=top_k * 2)  # Get more for reranking
        timing['hybrid_search'] = time.time() - hybrid_start
        print(f"✓ Hybrid search: {timing['hybrid_search']:.2f}s")
        
        # Step 2: Rerank
        rerank_start = time.time()
        documents = [result['document'] for result in hybrid_results]
        top_indices = self._rerank(query, documents, top_k=top_k)
        timing['rerank'] = time.time() - rerank_start
        print(f"✓ Reranking: {timing['rerank']:.2f}s")
        
        # Get reranked results
        reranked_results = [hybrid_results[i] for i in top_indices]
        
        # Step 3: Generate answer
        answer_start = time.time()
        context_docs = [result['document'] for result in reranked_results]
        answer = self._generate_answer(query, context_docs)
        timing['llm_answer'] = time.time() - answer_start
        print(f"✓ LLM answer generation: {timing['llm_answer']:.2f}s")
        
        timing['total'] = time.time() - pipeline_start
        print(f"✓ Total pipeline time: {timing['total']:.2f}s")
        print(f"  Breakdown: hybrid_search={timing['hybrid_search']:.2f}s, rerank={timing['rerank']:.2f}s, llm={timing['llm_answer']:.2f}s")
        
        return {
            'answer': answer,
            'sources': [
                {
                    'document': result['document'],
                    'metadata': result['metadata'],
                    'score': result['hybrid_score']
                }
                for result in reranked_results
            ],
            'query': query,
            'timing': timing  # Include timing breakdown in response
        }

# Global service instance
_rag_service = None

def get_rag_service() -> RAGService:
    """Get or create the RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def ask_question(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Backward-compatible helper used by the Streamlit frontend."""
    return get_rag_service().ask(query, top_k=top_k)
