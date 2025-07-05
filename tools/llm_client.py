import os
import requests
from typing import Optional, Dict, List
from dotenv import load_dotenv
import json

load_dotenv()

class LMStudioClient:
    def __init__(self):
        self.base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
        self.api_key = os.getenv("LM_STUDIO_API_KEY", "not-needed")
        self.model_name = os.getenv("MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct-gguf")
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         temperature: Optional[float] = None) -> str:
        """Generate response using LM Studio local server"""
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": 1000,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling LM Studio: {e}")
            return "I apologize, but I'm having trouble connecting to the local model. Please ensure LM Studio is running."
        except KeyError as e:
            print(f"Unexpected response format: {e}")
            return "I received an unexpected response format from the model."

# Test the client
if __name__ == "__main__":
    client = LMStudioClient()
    messages = [{"role": "user", "content": "Hello, are you working?"}]
    response = client.generate_response(messages)
    print(f"Response: {response}")