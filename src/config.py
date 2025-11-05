"""
Configuration management for RAG system.
Supports multiple profiles: docker, fast, synthetic.
"""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    """Base configuration class."""
    
    PROFILE_NAME: str = "default"
    
    # Paths
    BASE_DIR: str = os.path.expanduser("~/llm")
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    
    # Qdrant configuration
    QDRANT_URL: Optional[str] = None
    QDRANT_PATH: Optional[str] = None
    
    # Collection names
    COLL_RADLEX: str = "radlex_terms"
    COLL_LOINC: str = "loinc_procedures"
    COLL_RECIST: str = "recist_guidelines"
    COLL_PUBMED: str = "pubmed_abstracts"
    COLL_ABSTRACTS: str = "pubmed_abstracts"
    COLL_SYNTHETIC: str = "synthetic_reports"
    
    ALL_COLLECTIONS: List[str] = None
    
    # Model configuration
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANK_MODEL: str = "BAAI/bge-reranker-base"
    GEN_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Performance
    DEVICE: str = "cpu"
    UPLOAD_BATCH_SIZE: int = 128
    
    def __post_init__(self):
        if self.ALL_COLLECTIONS is None:
            self.ALL_COLLECTIONS = [
                self.COLL_RADLEX,
                self.COLL_LOINC,
                self.COLL_RECIST,
                self.COLL_PUBMED
            ]


@dataclass
class DockerConfig(Config):
    """Configuration for Qdrant in Docker with MiniLM embeddings."""
    
    PROFILE_NAME: str = "docker"
    
    # Paths
    BASE_DIR: str = os.path.expanduser("~/rad_rag_docker")
    HF_CACHE: str = os.path.join(BASE_DIR, "hf_cache")
    
    # Qdrant configuration
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_PATH: Optional[str] = None
    
    # Collection names (using MiniLM embeddings)
    COLL_RADLEX: str = "radlex_terms_minilm"
    COLL_LOINC: str = "loinc_procedures_minilm"
    COLL_RECIST: str = "recist_guidelines_minilm"
    COLL_PUBMED: str = "pubmed_abstracts_minilm"
    COLL_ABSTRACTS: str = "pubmed_abstracts_minilm"
    COLL_SYNTHETIC: str = "synthetic_reports_minilm"
    
    # Model configuration
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEVICE: str = "mps"  # Use Apple Silicon GPU if available
    UPLOAD_BATCH_SIZE: int = 128
    
    # Modal configuration for LLM inference
    MODAL_GPU_TYPE: str = "H100"  # H100 is faster for this model size
    MODAL_TIMEOUT: int = 300
    MODAL_IDLE_TIMEOUT: int = 120
    MODAL_APP_NAME: str = "radiology-rag"


def get_config(profile: str = 'docker') -> Config:
    """
    Get configuration for specified profile.
    
    Args:
        profile: Configuration profile ('docker', 'fast', 'synthetic', 'default')
        
    Returns:
        Config object with appropriate settings
    """
    profile = profile.lower()
    
    if profile == 'docker':
        return DockerConfig()
    elif profile == 'default':
        return Config()
    else:
        # Default to docker for now
        return DockerConfig()
