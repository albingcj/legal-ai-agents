from abc import ABC, abstractmethod
from typing import Dict, Any, List
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BaseAgent(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        self.logger = logging.getLogger(name)
        self.processing_history = []
        
    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state and return updated state"""
        pass
    
    def log(self, message: str, level: str = "INFO"):
        """Enhanced logging with different levels"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'agent': self.name,
            'level': level,
            'message': message
        }
        
        # Add to processing history
        self.processing_history.append(log_entry)
        
        # Log to console and file
        if level == "ERROR":
            self.logger.error(f"{self.name}: {message}")
        elif level == "WARNING":
            self.logger.warning(f"{self.name}: {message}")
        elif level == "DEBUG":
            self.logger.debug(f"{self.name}: {message}")
        else:
            self.logger.info(f"{self.name}: {message}")
        
        # Also print to console for immediate feedback
        print(f"[{timestamp}] {self.name} ({level}): {message}")
    
    def validate_state(self, state: Dict[str, Any], required_keys: List[str]) -> bool:
        """Enhanced state validation with detailed error messages"""
        missing_keys = []
        for key in required_keys:
            if key not in state:
                missing_keys.append(key)
        
        if missing_keys:
            self.log(f"Missing required keys: {missing_keys}", "ERROR")
            return False
        
        return True
    
    def get_processing_history(self) -> List[Dict]:
        """Get the processing history for this agent"""
        return self.processing_history.copy()
    
    def clear_history(self):
        """Clear the processing history"""
        self.processing_history.clear()
        self.log("Processing history cleared", "DEBUG")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information and statistics"""
        return {
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'total_processes': len(self.processing_history),
            'last_activity': self.processing_history[-1]['timestamp'] if self.processing_history else None
        }
