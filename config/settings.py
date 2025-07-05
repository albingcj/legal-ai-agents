import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application configuration settings"""
    
    # LM Studio Configuration
    LM_STUDIO_BASE_URL: str = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
    LM_STUDIO_API_KEY: str = os.getenv("LM_STUDIO_API_KEY", "not-needed")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct-gguf")
    
    # Vector Database
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Application Settings
    MAX_RETRIEVED_DOCS: int = int(os.getenv("MAX_RETRIEVED_DOCS", "8"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Legal Domain Settings
    SUPPORTED_DOMAINS: Dict[str, Dict[str, Any]] = {
        'contract_law': {
            'display_name': 'Contract Law',
            'description': 'Contract formation, performance, and breach',
            'keywords': ['contract', 'agreement', 'breach', 'consideration']
        },
        'employment_law': {
            'display_name': 'Employment Law',
            'description': 'Workplace rights and employment relationships',
            'keywords': ['employment', 'termination', 'discrimination', 'at-will']
        },
        'intellectual_property': {
            'display_name': 'Intellectual Property',
            'description': 'Copyright, patent, and trademark law',
            'keywords': ['copyright', 'patent', 'trademark', 'fair use']
        },
        'criminal_law': {
            'display_name': 'Criminal Law',
            'description': 'Criminal procedure and defense',
            'keywords': ['criminal', 'miranda', 'rights', 'defense']
        },
        'tort_law': {
            'display_name': 'Tort Law',
            'description': 'Civil liability and personal injury',
            'keywords': ['negligence', 'liability', 'damages', 'injury']
        }
    }
    
    # UI Settings
    STREAMLIT_CONFIG: Dict[str, Any] = {
        'page_title': 'Legal AI Assistant',
        'page_icon': '⚖️',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    }
    
    # Workflow Settings
    WORKFLOW_TIMEOUT: int = int(os.getenv("WORKFLOW_TIMEOUT", "300"))  # 5 minutes
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    
    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """Get all settings as dictionary"""
        return {
            'lm_studio': {
                'base_url': cls.LM_STUDIO_BASE_URL,
                'model_name': cls.MODEL_NAME
            },
            'vector_db': {
                'path': cls.CHROMA_DB_PATH,
                'embedding_model': cls.EMBEDDING_MODEL
            },
            'application': {
                'max_docs': cls.MAX_RETRIEVED_DOCS,
                'temperature': cls.TEMPERATURE,
                'debug': cls.DEBUG
            },
            'domains': cls.SUPPORTED_DOMAINS,
            'workflow': {
                'timeout': cls.WORKFLOW_TIMEOUT,
                'max_retries': cls.MAX_RETRIES
            }
        }

# Global settings instance
settings = Settings()
