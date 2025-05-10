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
try:
    from vertexai.generative_models import GenerativeModel, ChatSession
    from vertexai.generative_models import GenerationConfig, Content, Part
    from vertexai.preview.generative_models import GenerativeModel as PreviewGenerativeModel
    VERTEX_AI_AVAILABLE = True
except ImportError:
    logger.warning("Google Vertex AI SDK not available. Install with 'pip install google-cloud-aiplatform'")
    VERTEX_AI_AVAILABLE = False

class VertexAIService:
    """Service for interacting with Google Vertex AI."""
    
    def __init__(self):
        self.project_id = config.GCP_PROJECT_ID
        self.location = config.GCP_REGION
        self._models = {}  # Cache for GenerativeModel instances
        self._initialize()
    
    def _initialize(self):
        """Initialize Vertex AI service."""
        if not VERTEX_AI_AVAILABLE:
            logger.warning("Vertex AI SDK not available - skipping initialization")
            return
            
        if not self.project_id:
            logger.warning("GCP_PROJECT_ID not set - cannot initialize Vertex AI")
            return
            
        try:
            # Initialize Vertex AI
            import vertexai
            vertexai.init(project=self.project_id, location=self.location)
            logger.info(f"Initialized Vertex AI with project={self.project_id}, location={self.location}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {str(e)}")
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
vertex_ai_service = VertexAIService()
