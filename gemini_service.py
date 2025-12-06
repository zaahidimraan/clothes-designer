"""
Gemini API service for generating clothing design images.
"""

import os
from io import BytesIO
from PIL import Image

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class GeminiDesignService:
    """Service class for interacting with Gemini API."""
    
    def __init__(self, api_key: str):
        """Initialize the Gemini service."""
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai package not installed")
        
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash-exp-image-generation"
    
    def generate_design(self, prompt: str) -> tuple:
        """
        Generate a clothing design image based on the prompt.
        
        Returns:
            Tuple of (PIL Image or None, status message)
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["image", "text"],
                    temperature=0.8,
                )
            )
            
            # Extract image from response
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        image_data = part.inline_data.data
                        image = Image.open(BytesIO(image_data))
                        return image, "Design generated successfully!"
                    elif hasattr(part, 'text') and part.text:
                        return None, f"Model response: {part.text}"
            
            return None, "No image generated. Please try different options."
            
        except Exception as e:
            error_msg = str(e)
            if "API key" in error_msg:
                return None, "Invalid API key. Please check your Gemini API key."
            elif "quota" in error_msg.lower():
                return None, "API quota exceeded. Please try again later."
            else:
                return None, f"Error generating design: {error_msg}"


def test_api_connection(api_key: str) -> tuple:
    """
    Test if the API key is valid.
    
    Returns:
        Tuple of (success boolean, message)
    """
    if not GENAI_AVAILABLE:
        return False, "google-genai package not installed. Run: pip install google-genai"
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say 'Connected' in one word"
        )
        return True, "✅ API connection successful!"
    except Exception as e:
        return False, f"❌ API connection failed: {str(e)}"