"""
Task 1: Document Loading and Processing

Complete the DocumentLoader class to:
1. Load the provided doc.txt file
2. Split text into chunks for better processing
3. Create simple embeddings using TF-IDF approach
4. Store embeddings for similarity search

Requirements:
- Read doc.txt from the project directory
- Split into chunks of ~500 characters with overlap
- Create TF-IDF based embeddings for semantic search
- Implement cosine similarity for search
"""

import numpy as np
from typing import List, Tuple, Dict
import re
from collections import Counter
import math

class DocumentLoader:
    def __init__(self):
        """
        Initialize the document loader
        
        Task: Complete initialization
        - Initialize empty lists for chunks and embeddings
        - Initialize vocabulary for TF-IDF
        - Set up any other needed variables
        """
        self.chunks = []
        self.embeddings = []
        self.vocabulary = {}
        self.word_doc_count = Counter()
        self.num_chunks_total = 0
    
    def load_document(self, file_path: str = "doc.txt") -> str:
        """
        Load the document from file
        
        Task: Complete this method to:
        1. Read the doc.txt file with UTF-8 encoding
        2. Handle file not found errors gracefully
        3. Return the text content as string
        
        Args:
            file_path: Path to the document file
            
        Returns:
            String content of the document, empty string if error
        """
        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def split_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Task: Complete text chunking:
        1. Split text into chunks of approximately chunk_size characters
        2. Add overlap between chunks for better context
        3. Handle edge cases (empty text, short text)
        4. Return list of non-empty text chunks
        
        Args:
            text: Full document text
            chunk_size: Target size for each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []

        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

        chunks = []
        for paragraph in paragraphs:
            if len(paragraph) <= chunk_size:
                chunks.append(paragraph)
            else:
                start = 0
                while start < len(paragraph):
                    end = start + chunk_size
                    chunk = paragraph[start:end].strip()
                    if chunk:
                        chunks.append(chunk)
                    start += chunk_size - overlap

        return [c for c in chunks if c]

    def create_vocabulary(self, chunks: List[str]) -> Dict[str, int]:
        """
        Create vocabulary from all chunks for TF-IDF
        
        Task: Complete vocabulary creation:
        1. Extract all unique words from chunks
        2. Convert to lowercase and remove punctuation
        3. Return dictionary mapping word to index
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Dictionary mapping words to indices
        """
        all_words = set()
        for chunk in chunks:
            words = re.findall(r'[a-z]+', chunk.lower())
            all_words.update(words)

        vocab = {}
        for idx, word in enumerate(sorted(all_words)):
            vocab[word] = idx
        return vocab

    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Create TF-IDF embeddings for text chunks
        
        Task: Complete embedding creation:
        1. Create vocabulary if not exists
        2. Calculate TF-IDF vectors for each chunk
        3. Return numpy array of embeddings
        
        TF-IDF formula:
        - TF = (word count in chunk) / (total words in chunk)
        - IDF = log(total chunks / chunks containing word)
        - TF-IDF = TF * IDF
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Numpy array of embeddings (shape: [num_chunks, vocab_size])
        """
        if not self.vocabulary:
            self.vocabulary = self.create_vocabulary(chunks=chunks)
        vocab_size = len(self.vocabulary)
        num_chunks = len(chunks)
        embeddings = np.zeros((num_chunks, vocab_size))

        word_doc_count = Counter()
        for chunk in chunks:
            words = set(re.findall(r'[a-z]+', chunk.lower()))
            for word in words:
                if word in self.vocabulary:
                    word_doc_count[word] += 1

        self.word_doc_count = word_doc_count
        self.num_chunks_total = num_chunks

        for i, chunk in enumerate(chunks):
            words = set(re.findall(r'[a-z]+', chunk.lower()))
            total_words = len(words) if words else 1
            word_count = Counter(words)
            for word, count in word_count.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    tf = count / total_words
                    idf = math.log(num_chunks / (word_doc_count[word] + 1))
                    embeddings[i][idx] = idf * tf
        return embeddings

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Task: Implement cosine similarity:
        1. Calculate dot product of vectors
        2. Calculate magnitudes of both vectors
        3. Return cosine similarity (dot product / (mag1 * mag2))
        4. Handle zero division case
        
        Args:
            vec1, vec2: Numpy arrays representing vectors
            
        Returns:
            Cosine similarity score (0-1)
        """
        dot = np.dot(vec1, vec2)
        magnitude_1 = np.linalg.norm(vec1)
        magnitude_2 = np.linalg.norm(vec2)
        if magnitude_1 == 0 or magnitude_2 == 0:
            return 0.0
        return dot / (magnitude_1 * magnitude_2)
    
    def search_similar(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for similar chunks to the query
        
        Task: Complete similarity search:
        1. Convert query to TF-IDF vector using same vocabulary
        2. Calculate cosine similarity with all chunk embeddings
        3. Return top_k results sorted by similarity score
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of tuples (chunk_text, similarity_score)
        """
        if self.embeddings is None or not self.chunks:
            return []
        
        # converting query to TF using same vocabulary
        query_words = re.findall(r'[a-z]+', query.lower())
        total_words = len(query_words) if query_words else 1
        word_count = Counter(query_words)

        query_vector = np.zeros(len(self.vocabulary))
        for word, count in word_count.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                tf = count / total_words
                idf = math.log(self.num_chunks_total / (self.word_doc_count[word] + 1)) if self.word_doc_count[word] > 0 else 0
                query_vector[idx] = tf * idf
        
        similarities = []
        for i, chunk in enumerate(self.chunks):
            score = self.cosine_similarity(query_vector, self.embeddings[i])
            similarities.append((chunk, score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def load_and_process(self, file_path: str = "doc.txt") -> bool:
        """
        Complete pipeline to load and process the document
        
        Task: Complete the full processing pipeline:
        1. Load document from file_path
        2. Split into chunks
        3. Create embeddings
        4. Return True if successful, False otherwise
        
        Args:
            file_path: Path to document file
            
        Returns:
            Boolean indicating success
        """
        text = self.load_document(file_path=file_path)
        if not text:
            return False
        
        self.chunks = self.split_into_chunks(text)
        if not self.chunks:
            return False

        self.embeddings = self.create_embeddings(self.chunks)
        return True