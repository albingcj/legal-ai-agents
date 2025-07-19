import os
import time
from typing import Optional, Dict, List, Iterator, Any
from dotenv import load_dotenv
import litellm
from litellm import completion, acompletion

load_dotenv()

class UniversalLLMClient:
    """Universal LLM client supporting multiple providers via LiteLLM"""
    
    def __init__(self):
        # Load configuration from environment
        self.model = os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.api_base = os.getenv("LLM_API_BASE", "")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1000"))
        
        # LM Studio specific settings (for backward compatibility)
        lm_studio_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
        if lm_studio_url and (not self.model or self.model == "openai/gpt-3.5-turbo"):
            # If no specific model is set and LM Studio URL exists, use LM Studio
            self.model = "openai/local-model"  # LiteLLM format for local models
            self.api_base = f"{lm_studio_url}/v1"
            self.api_key = os.getenv("LM_STUDIO_API_KEY", "not-needed")
        
        # Set up LiteLLM configuration
        self._configure_litellm()
        
        # Store provider info for health checks
        self.provider_info = self._get_provider_info()
    
    def _configure_litellm(self):
        """Configure LiteLLM with environment variables"""
        # Set API key based on provider
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key
            os.environ["ANTHROPIC_API_KEY"] = self.api_key
            os.environ["GOOGLE_API_KEY"] = self.api_key
            os.environ["AZURE_API_KEY"] = self.api_key
        
        # Set API base if provided
        if self.api_base:
            os.environ["OPENAI_API_BASE"] = self.api_base
        
        # Configure LiteLLM settings
        litellm.set_verbose = False  # Set to True for debugging
        litellm.drop_params = True   # Drop unsupported parameters
        litellm.request_timeout = 30  # 30 second timeout
    
    def _get_provider_info(self) -> Dict[str, str]:
        """Extract provider information from model string"""
        parts = self.model.split('/')
        if len(parts) >= 2:
            provider = parts[0]
            model_name = '/'.join(parts[1:])
        else:
            provider = "unknown"
            model_name = self.model
        
        return {
            'provider': provider,
            'model_name': model_name,
            'full_model': self.model,
            'api_base': self.api_base
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the LLM API is responding and healthy"""
        try:
            # Test with a simple completion request
            test_messages = [{"role": "user", "content": "Say 'OK' if you're working correctly."}]
            
            start_time = time.time()
            response = completion(
                model=self.model,
                messages=test_messages,
                temperature=0.1,
                max_tokens=10,
                api_key=self.api_key,
                api_base=self.api_base if self.api_base else None
            )
            response_time = time.time() - start_time
            
            test_response = response.choices[0].message.content
            
            return {
                'status': 'healthy',
                'message': f'LLM API is responding correctly ({self.provider_info["provider"]})',
                'provider': self.provider_info['provider'],
                'model': self.provider_info['model_name'],
                'full_model': self.model,
                'api_base': self.api_base,
                'test_response': test_response.strip() if test_response else "",
                'response_time': round(response_time, 2),
                'usage': response.usage.model_dump() if hasattr(response, 'usage') and response.usage else None
            }
            
        except Exception as e:
            error_message = str(e)
            
            # Provide specific error messages based on common issues
            if "connection" in error_message.lower() or "refused" in error_message.lower():
                if self.provider_info['provider'] == 'openai' and 'localhost' in self.api_base:
                    specific_message = "Cannot connect to local LM Studio server. Please ensure LM Studio is running and the server is started."
                else:
                    specific_message = f"Cannot connect to {self.provider_info['provider']} API. Check your internet connection and API endpoint."
            elif "401" in error_message or "unauthorized" in error_message.lower():
                specific_message = f"Authentication failed for {self.provider_info['provider']}. Please check your API key."
            elif "timeout" in error_message.lower():
                specific_message = f"{self.provider_info['provider']} API is taking too long to respond. Server may be overloaded."
            elif "rate limit" in error_message.lower():
                specific_message = f"Rate limit exceeded for {self.provider_info['provider']} API. Please wait before trying again."
            elif "model" in error_message.lower() and "not found" in error_message.lower():
                specific_message = f"Model '{self.provider_info['model_name']}' not found or not accessible. Please check your model name."
            else:
                specific_message = f"LLM API error: {error_message}"
            
            return {
                'status': 'error',
                'message': specific_message,
                'provider': self.provider_info['provider'],
                'model': self.provider_info['model_name'],
                'full_model': self.model,
                'api_base': self.api_base,
                'error_details': error_message
            }
    
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         temperature: Optional[float] = None,
                         stream: bool = False) -> str:
        """Generate response using LiteLLM"""
        if stream:
            # For streaming, collect all chunks and return complete response
            full_response = ""
            for chunk in self.generate_response_stream(messages, temperature):
                full_response += chunk
            return full_response
        
        try:
            response = completion(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key if self.api_key else None,
                api_base=self.api_base if self.api_base else None
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e)
            
            # Return user-friendly error messages
            if "connection" in error_msg.lower():
                if 'localhost' in self.api_base:
                    return "I apologize, but I'm having trouble connecting to the local model. Please ensure LM Studio is running and the server is started."
                else:
                    return f"I apologize, but I'm having trouble connecting to {self.provider_info['provider']}. Please check your internet connection."
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                return f"Authentication failed with {self.provider_info['provider']}. Please check your API key configuration."
            elif "rate limit" in error_msg.lower():
                return f"Rate limit exceeded for {self.provider_info['provider']}. Please wait a moment before trying again."
            else:
                return f"I apologize, but I encountered an error: {error_msg}"
    
    def generate_response_stream(self, 
                               messages: List[Dict[str, str]], 
                               temperature: Optional[float] = None) -> Iterator[str]:
        """Generate streaming response using LiteLLM"""
        try:
            response = completion(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                api_key=self.api_key if self.api_key else None,
                api_base=self.api_base if self.api_base else None
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            error_msg = str(e)
            
            # Return user-friendly error messages for streaming
            if "connection" in error_msg.lower():
                if 'localhost' in self.api_base:
                    yield "I apologize, but I'm having trouble connecting to the local model. Please ensure LM Studio is running."
                else:
                    yield f"I apologize, but I'm having trouble connecting to {self.provider_info['provider']}."
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                yield f"Authentication failed with {self.provider_info['provider']}. Please check your API key."
            else:
                yield f"An error occurred during streaming: {error_msg}"
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get information about the current provider configuration"""
        return self.provider_info.copy()
    
    def get_supported_models(self) -> Dict[str, List[str]]:
        """Get list of supported models by provider"""
        return {
            "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
            "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-2.1", "claude-2.0", "claude-instant-1.2"],
            "google": ["gemini-pro", "gemini-pro-vision", "chat-bison", "text-bison"],
            "azure": ["azure/gpt-4", "azure/gpt-35-turbo"],
            "ollama": ["ollama/llama2", "ollama/codellama", "ollama/mistral"],
            "local": ["openai/local-model"]  # For LM Studio and other local servers
        }

# Maintain backward compatibility
class LMStudioClient(UniversalLLMClient):
    """Backward compatibility wrapper for LM Studio"""
    
    def __init__(self):
        # Override environment for LM Studio specific setup
        lm_studio_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
        os.environ["LLM_MODEL"] = "openai/local-model"
        os.environ["LLM_API_BASE"] = f"{lm_studio_url}/v1"
        os.environ["LLM_API_KEY"] = os.getenv("LM_STUDIO_API_KEY", "not-needed")
        
        super().__init__()

# Test the client
if __name__ == "__main__":
    print("Testing Universal LLM Client...")
    
    # Test with current configuration
    client = UniversalLLMClient()
    print(f"Provider: {client.provider_info}")
    
    # Health check
    print("\nTesting health check:")
    health = client.health_check()
    print(f"Health Status: {health}")
    
    if health['status'] == 'healthy':
        # Test regular response
        messages = [{"role": "user", "content": "Hello, are you working?"}]
        
        print("\nTesting regular response:")
        response = client.generate_response(messages)
        print(f"Response: {response}")
        
        print("\nTesting streaming response:")
        print("Streaming chunks: ", end="")
        for chunk in client.generate_response_stream(messages):
            print(chunk, end="", flush=True)
        print("\nStreaming complete!")
    else:
        print("Health check failed, skipping response tests.")
    
    print("\nSupported models:")
    models = client.get_supported_models()
    for provider, model_list in models.items():
        print(f"  {provider}: {model_list[:3]}...")  # Show first 3 models