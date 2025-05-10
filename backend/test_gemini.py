#!/usr/bin/env python
"""
Test script for Gemini 2.5 Flash Preview integration with custom API key.

This script tests the Gemini 2.5 Flash Preview integration by:
1. Loading the custom Gemini API key from the environment
2. Making a test query to the Gemini model
3. Displaying the response

Usage:
    python test_gemini.py
"""

import asyncio
import os
import sys
# No dotenv import needed as environment variables are loaded by the application
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after path setup
from services.llm import make_llm_api_call
from utils.config import config

async def test_gemini():
    """Test the Gemini 2.5 Flash Preview integration with a simple query."""
    
    # Check if GEMINI_API_KEY is set
    if not config.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set in .env file or environment variables.")
        logger.info("Please add GEMINI_API_KEY=your_api_key to your .env file")
        return False
    
    logger.info(f"Testing Gemini 2.5 Flash Preview integration with custom API key")
    
    # Prepare test messages
    test_messages = [
        {"role": "user", "content": "Hello! Can you confirm you're the Gemini 2.5 Flash Preview model and explain your key capabilities?"}
    ]

    try:
        # Make a test call to the Gemini model
        logger.info("Making test call to Gemini 2.5 Flash Preview...")
        response = await make_llm_api_call(
            model_name="gemini/gemini-2.5-flash-preview",
            messages=test_messages,
            temperature=0.7,
            max_tokens=500,
            api_key=config.GEMINI_API_KEY
        )
        
        # Print response details
        logger.info(f"✅ Successfully connected to Gemini API")
        logger.info(f"Model used: {response.model}")
        logger.info(f"Response content:\n{response.choices[0].message.content}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error testing Gemini: {str(e)}")
        return False

async def main():
    """Run all tests and report results."""
    
    # Run the Gemini test
    success = await test_gemini()
    
    if success:
        logger.info("\n✅ All tests completed successfully!")
    else:
        logger.error("\n❌ Tests failed. Please check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
