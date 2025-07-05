from abc import ABC, abstractmethod
from typing import Dict, Any, List
import json
from datetime import datetime

class BaseAgent(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        
    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state and return updated state"""
        pass
    
    def log(self, message: str, level: str = "INFO"):
        """Simple logging"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {self.name} ({level}): {message}")
    
    def validate_state(self, state: Dict[str, Any], required_keys: List[str]) -> bool:
        """Validate that required keys exist in state"""
        for key in required_keys:
            if key not in state:
                self.log(f"Missing required key: {key}", "ERROR")
                return False
        return True