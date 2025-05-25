"""
Document Store Module for RAG (Retrieval-Augmented Generation).
Implements ChromaDB-based storage for environmental data.
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class DocumentStore:
    """
    ChromaDB-based document store for environmental data and analysis results.
    Provides vector storage and semantic search capabilities.
    """
    
    def __init__(self, persist_directory: str = "data/chroma"):
        # Initialize ChromaDB client
        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            )
        )
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get collections
        self.env_collection = self.client.get_or_create_collection(
            name="environmental_data",
            embedding_function=self.embedding_function
        )
        
        self.analysis_collection = self.client.get_or_create_collection(
            name="analysis_results",
            embedding_function=self.embedding_function
        )
    
    def add_document(
        self,
        document: Dict[str, Any],
        collection_name: str = "environmental_data"
    ) -> str:
        """
        Add a document to the store.
        
        Args:
            document: Document to store
            collection_name: Name of collection to store in
            
        Returns:
            Document ID
        """
        # Generate document ID
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(document)}"
        
        # Prepare document metadata
        metadata = {
            'timestamp': document.get('timestamp', datetime.now().isoformat()),
            'type': document.get('type', 'general'),
            'source': document.get('source', 'unknown')
        }
        
        # Convert document to text for embedding
        doc_text = self._document_to_text(document)
        
        # Add to appropriate collection
        collection = self._get_collection(collection_name)
        collection.add(
            documents=[doc_text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        return doc_id
    
    def search(
        self,
        query: str,
        filters: Dict = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            filters: Optional filters
            top_k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        try:
            # Convert date range filters to single operator
            if filters and 'timestamp' in filters:
                if isinstance(filters['timestamp'], dict):
                    # Use only the upper bound for filtering
                    filters['timestamp'] = {'$lte': filters['timestamp'].get('$lte')}
            
            # Search environmental collection
            env_results = self.env_collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filters
            )
            
            # Format results
            results = []
            for i in range(len(env_results['ids'][0])):
                doc = {
                    'id': env_results['ids'][0][i],
                    'content': env_results['documents'][0][i],
                    'metadata': {
                        k: env_results['metadatas'][0][i][k]
                        for k in env_results['metadatas'][0][i]
                    },
                    'score': float(env_results['distances'][0][i])
                }
                results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"Error in document search: {e}")
            return []
    
    def _get_collection(self, name: str):
        """Get collection by name"""
        if name == "environmental_data":
            return self.env_collection
        elif name == "analysis_results":
            return self.analysis_collection
        else:
            raise ValueError(f"Unknown collection: {name}")
    
    def _document_to_text(self, document: Dict[str, Any]) -> str:
        """
        Convert document to text format for embedding.
        """
        text_parts = []
        
        # Add document type if available
        if 'type' in document:
            text_parts.append(f"Type: {document['type']}")
        
        # Add location if available
        if 'location' in document:
            loc = document['location']
            text_parts.append(
                f"Location: lat={loc.get('lat', 'N/A')}, "
                f"lon={loc.get('lon', 'N/A')}"
            )
        
        # Add environmental data
        if 'environmental_data' in document:
            env_data = document['environmental_data']
            text_parts.append("Environmental Data:")
            for key, value in env_data.items():
                text_parts.append(f"- {key}: {value}")
        
        # Add analysis results
        if 'analysis' in document:
            analysis = document['analysis']
            text_parts.append("Analysis Results:")
            for key, value in analysis.items():
                text_parts.append(f"- {key}: {value}")
        
        # Add any other relevant fields
        for key, value in document.items():
            if key not in ['type', 'location', 'environmental_data', 'analysis']:
                text_parts.append(f"{key}: {value}")
        
        return "\n".join(text_parts)
    
    def _text_to_document(self, text: str) -> Dict[str, Any]:
        """
        Convert text format back to document structure.
        """
        document = {}
        current_section = None
        
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if line.endswith(":"):
                # Start new section
                current_section = line[:-1].lower().replace(" ", "_")
                document[current_section] = {}
            elif line.startswith("- "):
                # Add to current section
                if current_section:
                    key, value = line[2:].split(": ", 1)
                    document[current_section][key] = value
            else:
                # Add top-level field
                if ": " in line:
                    key, value = line.split(": ", 1)
                    document[key.lower()] = value
        
        return document
    
    def save(self) -> None:
        """
        Persist collections to disk.
        """
        self.client.persist()
    
    def load(self) -> None:
        """
        Load collections from disk.
        """
        # ChromaDB automatically loads from persist_directory
        pass 