"""
Vertex AI integration for AgentPress.

This module provides a unified interface for making API calls to Google Vertex AI
to leverage Gemini models for text generation. It integrates with the existing LLM
service architecture to provide a seamless experience.
"""

import os
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
import asyncio
import json

from utils.logger import logger
from utils.config import config

# Check if we're running in an environment where Vertex AI SDK is available
VERTEX_AI_AVAILABLE = False
try:
    from vertexai.generative_models import GenerativeModel, ChatSession
    from vertexai.generative_models import GenerationConfig, Content, Part
    from vertexai.preview.generative_models import GenerativeModel as PreviewGenerativeModel
    VERTEX_AI_AVAILABLE = True
except ImportError:
    logger.warning("Google Vertex AI SDK not available. Install with 'pip install google-cloud-aiplatform'")
    # Define placeholder classes to avoid NameErrors when the module is imported
    class GenerativeModel:
        pass
    class ChatSession:
        pass
    class GenerationConfig:
        pass
    class Content:
        pass
    class Part:
        @classmethod
        def from_text(cls, text):
            return text
    class PreviewGenerativeModel(GenerativeModel):
        pass

class VertexAIService:
    """Service for interacting with Google Vertex AI.
    This class provides methods for generating content using Vertex AI models.
    """
    
    def __init__(self):
        # Import here to avoid errors if module is not available
        try:
            import vertexai
            import google.cloud.aiplatform as aiplatform
            from vertexai.generative_models import GenerativeModel, GenerationConfig
            from google.oauth2 import service_account
            import json
            import os
            
            self.vertexai = vertexai
            self.aiplatform = aiplatform
            self.GenerativeModel = GenerativeModel
            self.GenerationConfig = GenerationConfig
            self.service_account = service_account
            self.initialized = True
            self.json = json
            self.os = os
            
            # Setup verification to ensure credentials are valid
            self._check_and_setup_credentials()
            
            self.project_id = config.GCP_PROJECT_ID
            self.location = config.GCP_REGION
            self._models = {}  # Cache for GenerativeModel instances
            self._initialize()
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Unable to import Vertex AI modules, service will be unavailable: {str(e)}")
            self.initialized = False
            
    def _check_and_setup_credentials(self):
        """Verify that credentials are available and set them up if possible.
        This method tries different approaches to ensure Vertex AI can authenticate.
        """
        if not self.initialized:
            logger.warning("VertexAI service not initialized, cannot set up credentials")
            return False
            
        # 1. Check if GOOGLE_APPLICATION_CREDENTIALS environment variable is set
        creds_path = self.os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if creds_path and self.os.path.exists(creds_path):
            logger.info(f"Using Google credentials from: {creds_path}")
            return True
            
        # 2. Check for credentials in standard locations
        # Try standard locations where gcloud credentials might be stored
        default_locations = [
            "~/.config/gcloud/application_default_credentials.json",
            "~/.gcloud/application_default_credentials.json"
        ]
        
        for loc in default_locations:
            expanded_path = self.os.path.expanduser(loc)
            if self.os.path.exists(expanded_path):
                self.os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = expanded_path
                logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS to found credentials at: {expanded_path}")
                return True
                
        # 3. Check for credentials in configuration
        if config.GCP_SERVICE_ACCOUNT_JSON:
            try:
                # Create a temporary file with the credentials
                import tempfile
                credentials_dict = self.json.loads(config.GCP_SERVICE_ACCOUNT_JSON)
                
                # Write the credentials to a temporary file
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
                    self.json.dump(credentials_dict, temp_file)
                    temp_path = temp_file.name
                
                # Set the environment variable to point to the credentials file
                self.os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_path
                logger.info(f"Created temporary credentials file at: {temp_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to set up credentials from GCP_SERVICE_ACCOUNT_JSON: {str(e)}")
        
        logger.warning("No Google Cloud credentials found or set up. Vertex AI calls may fail.")
        return False
            
    def _initialize(self):
        """Initialize Vertex AI service."""
        if not VERTEX_AI_AVAILABLE:
            logger.warning("Vertex AI modules not available, service not initialized.")
            self.initialized = False
            return
            
        if not self.project_id:
            logger.error("GCP_PROJECT_ID is not set. Cannot initialize Vertex AI.")
            self.initialized = False
            return
            
        if not self.location:
            logger.warning("GCP_REGION is not set. Using default value: us-central1")
            self.location = "us-central1"
            
        try:
            # Try to initialize with credentials from service account JSON if available
            if config.GCP_SERVICE_ACCOUNT_JSON:
                try:
                    credentials_dict = self.json.loads(config.GCP_SERVICE_ACCOUNT_JSON)
                    credentials = self.service_account.Credentials.from_service_account_info(credentials_dict)
                    logger.info("Using credentials from GCP_SERVICE_ACCOUNT_JSON")
                    
                    # Initialize Vertex AI with explicit credentials
                    self.vertexai.init(
                        project=self.project_id, 
                        location=self.location,
                        credentials=credentials
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize Vertex AI with service account JSON: {str(e)}")
                    logger.info("Falling back to application default credentials")
                    self.vertexai.init(project=self.project_id, location=self.location)
            else:
                # Initialize with application default credentials
                self.vertexai.init(project=self.project_id, location=self.location)
                
            logger.info(f"Successfully initialized Vertex AI with project={self.project_id}, location={self.location}")
            
            # Test connection by listing available models
            try:
                self._test_connection()
                logger.info("Vertex AI connection successfully verified")
            except Exception as e:
                logger.warning(f"Vertex AI initialized but connection test failed: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {str(e)}")
            self.initialized = False
            
    def _test_connection(self):
        """Test the connection to Vertex AI by listing models."""
        if not self.initialized:
            raise RuntimeError("Service not initialized")
            
        try:
            # Perform a simple operation to verify connectivity
            publisher_models = self.aiplatform.PublisherModel.list()
            model_count = len(list(publisher_models))
            logger.info(f"Successfully connected to Vertex AI. Found {model_count} publisher models.")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            raise
    
    def get_model(self, model_name: str) -> Union[GenerativeModel, None]:
        """Get or create a GenerativeModel instance for the given model name."""
        if not VERTEX_AI_AVAILABLE:
            logger.error("Vertex AI SDK not available")
            return None
            
        # Parse out the model name from our format (vertex/gemini-1.5-pro -> gemini-1.5-pro)
        if model_name.startswith("vertex/"):
            vertex_model_name = model_name.split("/", 1)[1]
        else:
            vertex_model_name = model_name
            
        # Return from cache if available
        if vertex_model_name in self._models:
            return self._models[vertex_model_name]
            
        # Create new model instance
        try:
            # For preview models, use the preview class
            if "preview" in vertex_model_name:
                model = PreviewGenerativeModel(vertex_model_name)
            else:
                model = GenerativeModel(vertex_model_name)
            self._models[vertex_model_name] = model
            logger.info(f"Created Vertex AI model: {vertex_model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to create Vertex AI model {vertex_model_name}: {str(e)}")
            return None
    
    def _convert_openai_messages_to_vertex(self, messages: List[Dict[str, Any]]) -> List[Content]:
        """Convert OpenAI-style messages to Vertex AI content format."""
        vertex_messages = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Map OpenAI roles to Vertex AI roles
            if role == "system":
                role = "user"  # Vertex doesn't have system role, prepend to first user message
            elif role == "assistant":
                role = "model"
            
            vertex_messages.append(Content(role=role, parts=[Part.from_text(content)]))
            
        return vertex_messages
    
    async def generate_content(
        self, 
        model_name: str, 
        messages: List[Dict[str, Any]], 
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Generate content using a Vertex AI model."""
        if not VERTEX_AI_AVAILABLE:
            raise RuntimeError("Vertex AI SDK not available")
            
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Failed to get Vertex AI model: {model_name}")
            
        # Convert OpenAI format messages to Vertex AI format
        vertex_messages = self._convert_openai_messages_to_vertex(messages)
        
        # Prepare generation config
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens if max_tokens else None,
            top_p=kwargs.get("top_p", 0.95),
        )
        
        try:
            if stream:
                # For streaming, we'll use an async generator
                return self._stream_content(model, vertex_messages, generation_config)
            else:
                # For non-streaming, just return the response
                response = model.generate_content(
                    contents=vertex_messages,
                    generation_config=generation_config
                )
                
                # Convert to OpenAI-like format for compatibility
                return self._convert_vertex_response_to_openai(response)
                
        except Exception as e:
            logger.error(f"Error generating content with Vertex AI: {str(e)}")
            raise
    
    async def _stream_content(self, model, vertex_messages, generation_config):
        """Stream content from a Vertex AI model."""
        try:
            response_stream = model.generate_content(
                contents=vertex_messages,
                generation_config=generation_config,
                stream=True
            )
            
            # Create OpenAI-like streaming format
            for response in response_stream:
                yield self._convert_vertex_chunk_to_openai(response)
                
        except Exception as e:
            logger.error(f"Error streaming content from Vertex AI: {str(e)}")
            raise
    
    def _convert_vertex_response_to_openai(self, response) -> Dict[str, Any]:
        """Convert Vertex AI response to OpenAI-compatible format."""
        try:
            text_content = response.text
            
            # Create OpenAI-like response structure
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text_content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "model": response.candidates[0].citation_metadata.citation_sources[0].uri if hasattr(response, 'candidates') and hasattr(response.candidates[0], 'citation_metadata') else "vertex-ai",
                "object": "chat.completion",
                "created": None,  # No timestamp in Vertex response
                "id": None,  # No id in Vertex response
            }
        except Exception as e:
            logger.error(f"Error converting Vertex AI response to OpenAI format: {str(e)}")
            # Return minimal valid response
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": str(e)
                        },
                        "finish_reason": "stop"
                    }
                ],
                "model": "vertex-ai",
                "object": "chat.completion",
            }
    
    def _convert_vertex_chunk_to_openai(self, chunk) -> Dict[str, Any]:
        """Convert Vertex AI streaming chunk to OpenAI-compatible format."""
        try:
            # Get the delta text from the chunk
            chunk_text = chunk.text
            
            # Create OpenAI-like chunk structure
            return {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": chunk_text
                        },
                        "finish_reason": None
                    }
                ],
                "model": "vertex-ai",
                "object": "chat.completion.chunk",
            }
        except Exception as e:
            logger.error(f"Error converting Vertex AI chunk to OpenAI format: {str(e)}")
            # Return minimal valid chunk
            return {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": ""
                        },
                        "finish_reason": "stop"
                    }
                ],
                "model": "vertex-ai",
                "object": "chat.completion.chunk",
            }

# Create a singleton instance
if VERTEX_AI_AVAILABLE:
    try:
        vertex_ai_service = VertexAIService()
        logger.info("Vertex AI service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI service: {str(e)}")
        # Create a dummy service that will gracefully fail if used
        vertex_ai_service = None
else:
    logger.warning("Vertex AI SDK not available - creating a placeholder service")
    # Create a dummy service
    vertex_ai_service = VertexAIService()
