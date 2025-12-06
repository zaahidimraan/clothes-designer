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
        # Updated model name for image generation
        self.image_model = "imagen-3.0-generate-002"
        self.text_model = "gemini-2.0-flash"
    
    def generate_design(self, prompt: str) -> tuple:
        """
        Generate a clothing design image based on the prompt.
        
        Returns:
            Tuple of (PIL Image or None, status message)
        """
        try:
            # Use Imagen 3 for image generation
            response = self.client.models.generate_images(
                model=self.image_model,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="1:1",
                    safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
                )
            )
            
            # Extract image from response
            if response.generated_images:
                image_data = response.generated_images[0].image.image_bytes
                image = Image.open(BytesIO(image_data))
                return image, "Design generated successfully!"
            
            return None, "No image generated. Please try different options."
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if Imagen is not available, try alternative approach
            if "not found" in error_msg.lower() or "not supported" in error_msg.lower():
                return self._generate_with_gemini_flash(prompt)
            elif "API key" in error_msg:
                return None, "Invalid API key. Please check your Gemini API key."
            elif "quota" in error_msg.lower():
                return None, "API quota exceeded. Please try again later."
            elif "safety" in error_msg.lower():
                return None, "Content was blocked by safety filters. Try different options."
            else:
                return None, f"Error generating design: {error_msg}"
    
    def _generate_with_gemini_flash(self, prompt: str) -> tuple:
        """
        Alternative: Generate using Gemini 2.0 Flash with image output.
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
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
                        # Only text returned, no image
                        return None, f"Model couldn't generate image. Response: {part.text[:200]}"
            
            return None, "No image generated. Please try different options."
            
        except Exception as e:
            return None, f"Error with alternative model: {str(e)}"


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
        
        # Test with text model first
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say 'Connected' in one word"
        )
        
        # Check available models for image generation
        models_info = check_available_models(client)
        
        return True, f"✅ API connection successful!\n{models_info}"
    except Exception as e:
        return False, f"❌ API connection failed: {str(e)}"


def check_available_models(client) -> str:
    """Check which image generation models are available."""
    try:
        models = []
        for model in client.models.list():
            model_name = model.name if hasattr(model, 'name') else str(model)
            if 'imagen' in model_name.lower() or 'image' in model_name.lower():
                models.append(model_name)
        
        if models:
            return f"Available image models: {', '.join(models[:3])}"
        return "Using Gemini Flash for image generation"
    except:
        return "Using default image generation"