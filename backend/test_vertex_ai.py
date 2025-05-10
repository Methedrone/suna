"""
Test script for Vertex AI integration.

This script demonstrates and tests the integration with Google Vertex AI.
It verifies that the Vertex AI services can be initialized and used to generate content.
"""

import os
import asyncio
import logging
import sys

# Próba importu dotenv, ale kontynuuj nawet jeśli nie jest dostępny
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Załadowano zmienne środowiskowe z pliku .env")
except ImportError:
    print("Uwaga: Moduł dotenv nie jest zainstalowany. Korzystam z istniejących zmiennych środowiskowych.")
    # Funkcja zastępcza, żeby kod działał dalej
    def load_dotenv():
        print("Pominięto ładowanie .env (brak modułu dotenv)")
        pass

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import after environment variables are loaded
from backend.utils.config import config
from backend.services.vertex_ai import vertex_ai_service, VERTEX_AI_AVAILABLE

async def test_vertex_ai_direct():
    """Test the Vertex AI service directly."""
    if not VERTEX_AI_AVAILABLE:
        logger.error("Vertex AI SDK not available. Please install with 'pip install google-cloud-aiplatform'")
        return False

    if not config.GCP_PROJECT_ID:
        logger.error("GCP_PROJECT_ID is not set. Please set it in your .env file")
        return False

    # Enable Vertex AI for testing
    config.VERTEX_AI_ENABLED = True
    
    logger.info(f"Testing Vertex AI with project ID: {config.GCP_PROJECT_ID} in region: {config.GCP_REGION}")
    
    # Test messages
    messages = [
        {"role": "user", "content": "Hello, please introduce yourself briefly."}
    ]
    
    try:
        # Test with Gemini 1.5 Flash model
        model_name = "vertex/gemini-1.5-flash"
        logger.info(f"Testing Vertex AI with model: {model_name}")
        
        response = await vertex_ai_service.generate_content(
            model_name=model_name,
            messages=messages,
            temperature=0.2
        )
        
        if response and response.get("choices") and len(response["choices"]) > 0:
            logger.info("✅ Vertex AI test successful!")
            message_content = response["choices"][0]["message"]["content"]
            logger.info(f"Response: {message_content[:200]}...")
            return True
        else:
            logger.error(f"❌ Unexpected response format: {response}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing Vertex AI: {str(e)}")
        return False

async def test_vertex_ai_via_llm_service():
    """Test the Vertex AI integration through the LLM service."""
    from backend.services.llm import make_llm_api_call
    
    if not VERTEX_AI_AVAILABLE:
        logger.error("Vertex AI SDK not available. Please install with 'pip install google-cloud-aiplatform'")
        return False

    if not config.GCP_PROJECT_ID:
        logger.error("GCP_PROJECT_ID is not set. Please set it in your .env file")
        return False

    # Enable Vertex AI for testing
    config.VERTEX_AI_ENABLED = True
    
    logger.info(f"Testing Vertex AI via LLM service with project ID: {config.GCP_PROJECT_ID}")
    
    # Test messages
    messages = [
        {"role": "user", "content": "What are three advantages of using Vertex AI?"}
    ]
    
    try:
        # Test with Gemini 1.5 Flash model through the LLM service
        model_name = "vertex/gemini-1.5-flash"
        logger.info(f"Testing LLM service with Vertex AI model: {model_name}")
        
        response = await make_llm_api_call(
            model_name=model_name,
            messages=messages,
            temperature=0.2
        )
        
        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            logger.info("✅ Vertex AI via LLM service test successful!")
            message_content = response.choices[0].message.content
            logger.info(f"Response: {message_content[:200]}...")
            return True
        else:
            logger.error(f"❌ Unexpected response format: {response}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing Vertex AI via LLM service: {str(e)}")
        return False

async def run_tests():
    """Run all Vertex AI tests."""
    logger.info("Starting Vertex AI integration tests")
    
    # Set environment variables for testing if not already set
    if not config.GCP_PROJECT_ID:
        # For testing, you can set a default project ID here
        config.GCP_PROJECT_ID = "your-project-id"
        logger.warning(f"Setting default GCP_PROJECT_ID for testing: {config.GCP_PROJECT_ID}")
    
    direct_test_result = await test_vertex_ai_direct()
    logger.info(f"Direct Vertex AI test result: {'✅ PASSED' if direct_test_result else '❌ FAILED'}")
    
    llm_service_test_result = await test_vertex_ai_via_llm_service()
    logger.info(f"LLM service Vertex AI test result: {'✅ PASSED' if llm_service_test_result else '❌ FAILED'}")
    
    return direct_test_result and llm_service_test_result

if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_tests())
