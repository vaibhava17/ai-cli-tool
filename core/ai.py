"""
AI Client - Multi-Provider AI Interface with Intelligent Routing

This module provides a unified interface for interacting with multiple AI providers
through LiteLLM. It handles provider selection, task categorization, API configuration,
and response processing with automatic fallback strategies.

Key Features:
- Multi-provider support (OpenAI, Anthropic, Gemini, Ollama)
- Intelligent task categorization and provider routing
- Dynamic provider/model override capabilities
- Multimodal support (text, images, files)
- Comprehensive error handling and fallback strategies
- Automatic API key management and environment setup

The AIClient class serves as the central orchestrator for all AI interactions,
abstracting away provider-specific implementations while maintaining full
flexibility for advanced use cases.

Dependencies:
- litellm: Universal API client for multiple AI providers
- config.setting: Configuration management and provider definitions
- typing: Type hints for better code documentation
- logging: Structured logging for debugging and monitoring

Author: AI CLI Tool Team
License: MIT
"""

import litellm
import os
from typing import Optional, List, Dict, Any
from config.setting import AI_PROVIDERS, TASK_CATEGORIES
from config.logger import get_logger, log_ai_request

logger = get_logger(__name__)

class AIClient:
    def __init__(self):
        self._validate_providers()
        self._setup_litellm()
    
    def _validate_providers(self):
        """Validate that at least one provider is properly configured"""
        valid_providers = []
        for category, config in AI_PROVIDERS.items():
            if config.get("api_key"):
                valid_providers.append(category)
        
        if not valid_providers:
            raise RuntimeError("No AI providers configured. Please set API keys in .env file")
        
        logger.info(f"Available AI providers: {valid_providers}")
    
    def _setup_litellm(self):
        """Configure litellm with provider settings"""
        from config.setting import AVAILABLE_PROVIDERS
        
        # Configure litellm settings
        litellm.set_verbose = False
        litellm.drop_params = True  # Drop unsupported parameters automatically
        
        # Set API keys for all available providers
        for provider, config in AVAILABLE_PROVIDERS.items():
            api_key_env = config.get("api_key_env")
            if api_key_env and os.getenv(api_key_env):
                if provider == "openai":
                    os.environ["OPENAI_API_KEY"] = os.getenv(api_key_env)
                elif provider == "anthropic":
                    os.environ["ANTHROPIC_API_KEY"] = os.getenv(api_key_env)
                elif provider == "google":
                    os.environ["GEMINI_API_KEY"] = os.getenv(api_key_env)
                
                logger.info(f"Configured {provider} provider")
        
        # Configure additional litellm settings for better error handling
        litellm.suppress_debug_info = True
        
        # Set default timeout and retry settings
        litellm.request_timeout = 60
        litellm.num_retries = 2
    
    def _categorize_task(self, prompt: str) -> str:
        """Categorize task based on prompt content"""
        prompt_lower = prompt.lower()
        
        for category, keywords in TASK_CATEGORIES.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return category
        
        return "general"
    
    def _get_provider_config(self, category: str, override_provider: Optional[str] = None, override_model: Optional[str] = None) -> Dict[str, Any]:
        """Get provider configuration for a specific category with optional overrides"""
        config = AI_PROVIDERS.get(category, AI_PROVIDERS["general"]).copy()
        
        # Override provider/model from CLI arguments
        if override_provider:
            from config.setting import AVAILABLE_PROVIDERS, get_api_key_for_provider
            
            if override_provider in AVAILABLE_PROVIDERS:
                config["provider"] = override_provider
                config["api_key"] = get_api_key_for_provider(override_provider)
                
                # Set default model if not specified
                if not override_model:
                    available_models = AVAILABLE_PROVIDERS[override_provider]["models"]
                    if available_models:
                        config["model"] = available_models[0]  # Use first available model
                else:
                    config["model"] = override_model
            else:
                logger.warning(f"Unknown provider: {override_provider}. Using default.")
        elif override_model:
            config["model"] = override_model
        
        # Fallback to general if category provider not configured
        if not config.get("api_key"):
            config = AI_PROVIDERS["general"].copy()
        
        # Final fallback to any configured provider
        if not config.get("api_key"):
            for cat, cfg in AI_PROVIDERS.items():
                if cfg.get("api_key"):
                    config = cfg.copy()
                    break
        
        return config
    
    def chat(
        self, 
        prompt: str, 
        context: Optional[str] = None, 
        files: Optional[List[bytes]] = None, 
        images: Optional[List[bytes]] = None,
        category: Optional[str] = None,
        override_provider: Optional[str] = None,
        override_model: Optional[str] = None
    ) -> str:
        """Send chat request to appropriate AI provider based on task category"""
        
        # Auto-categorize if not specified
        if not category:
            category = self._categorize_task(prompt)
        
        config = self._get_provider_config(category, override_provider, override_model)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use provided context if relevant."}
        ]
        
        # Add context if provided
        if context:
            user_content = f"Context:\n{context}\n\nUser Query:\n{prompt}"
        else:
            user_content = prompt
        
        # Handle multimodal content
        if images or files:
            content = [{"type": "text", "text": user_content}]
            
            # Add images
            if images:
                for img_bytes in images:
                    import base64
                    img_b64 = base64.b64encode(img_bytes).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })
            
            # Add file content (as text for now)
            if files:
                for file_bytes in files:
                    try:
                        file_text = file_bytes.decode('utf-8', errors='ignore')[:2000]  # Limit size
                        content.append({
                            "type": "text", 
                            "text": f"\nFile content:\n{file_text}"
                        })
                    except Exception as e:
                        logger.warning(f"Could not process file: {e}")
            
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_content})
        
        try:
            # Call litellm with appropriate model
            model_name = f"{config['provider']}/{config['model']}"
            
            # Log AI request
            log_ai_request(config['provider'], config['model'], category)
            
            response = litellm.completion(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=4000,
            )
            
            logger.info(f"Used {category} provider: {model_name}")
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling AI provider {config['provider']}: {e}")
            raise RuntimeError(f"AI request failed: {str(e)}")