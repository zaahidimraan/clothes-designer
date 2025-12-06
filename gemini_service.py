"""
Gemini API service for generating clothing design images.
With fallback options and detailed error messages.
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


# Models to try in order (based on your available models)
IMAGE_MODELS = [
    "gemini-2.5-flash-preview-image-generation",
    "models/gemini-2.5-flash-image-preview",
    "models/gemini-2.5-flash-image",
    "models/gemini-3-pro-image-preview",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.0-flash-exp",
]


class GeminiDesignService:
    """Service class for interacting with Gemini API."""
    
    def __init__(self, api_key: str):
        """Initialize the Gemini service."""
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai package not installed")
        
        self.client = genai.Client(api_key=api_key)
        self.available_models = self._get_available_image_models()
    
    def _get_available_image_models(self) -> list:
        """Get list of available image generation models."""
        try:
            models = []
            for model in self.client.models.list():
                name = model.name if hasattr(model, 'name') else str(model)
                if 'image' in name.lower():
                    models.append(name)
            return models if models else IMAGE_MODELS
        except:
            return IMAGE_MODELS
    
    def generate_design(self, prompt: str) -> tuple:
        """
        Generate a clothing design image based on the prompt.
        Tries multiple models until one works.
        
        Returns:
            Tuple of (PIL Image or None, status message)
        """
        
        # Combine available models with default list
        models_to_try = self.available_models + IMAGE_MODELS
        # Remove duplicates while preserving order
        models_to_try = list(dict.fromkeys(models_to_try))
        
        errors = []
        country_blocked = False
        
        for model_name in models_to_try:
            try:
                # Clean model name (remove 'models/' prefix if needed for some calls)
                clean_name = model_name.replace("models/", "") if model_name.startswith("models/") else model_name
                
                response = self.client.models.generate_content(
                    model=clean_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["image", "text"],
                        temperature=0.9,
                    )
                )
                
                # Extract image from response
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            image_data = part.inline_data.data
                            image = Image.open(BytesIO(image_data))
                            return image, f"✅ Design generated successfully!\n(Model: {clean_name})"
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for country restriction
                if "not available in your country" in error_msg or "failed_precondition" in error_msg:
                    country_blocked = True
                    errors.append(f"{model_name}: Country restricted")
                elif "not found" in error_msg:
                    errors.append(f"{model_name}: Not found")
                elif "quota" in error_msg:
                    errors.append(f"{model_name}: Quota exceeded")
                else:
                    errors.append(f"{model_name}: {str(e)[:50]}")
                
                continue
        
        # All models failed - return appropriate error
        if country_blocked:
            return None, self._get_country_blocked_message()
        
        return None, f"❌ All models failed:\n" + "\n".join(errors[:3])
    
    def _get_country_blocked_message(self) -> str:
        """Return detailed message when country is blocked."""
        return """
🌍 **Image Generation Not Available in Your Region**

Google's Gemini image generation is currently restricted in your country/region.

**Alternative Solutions:**

1. **Use a VPN** - Connect to US, UK, or EU servers

2. **Use Alternative APIs:**
   - **Stability AI** (Stable Diffusion): https://stability.ai/
   - **OpenAI DALL-E**: https://openai.com/dall-e
   - **Replicate**: https://replicate.com/
   - **Hugging Face**: https://huggingface.co/

3. **Free Alternatives:**
   - **Leonardo.AI**: https://leonardo.ai/ (Free tier available)
   - **Playground AI**: https://playground.ai/ (Free tier)
   - **Bing Image Creator**: https://www.bing.com/create (Free)

4. **Self-hosted Options:**
   - Run Stable Diffusion locally
   - Use Google Colab with SD models

Would you like me to help integrate an alternative API?
"""


def test_api_connection(api_key: str) -> tuple:
    """
    Test if the API key is valid and check image generation availability.
    """
    if not GENAI_AVAILABLE:
        return False, "❌ google-genai package not installed. Run: pip install google-genai"
    
    try:
        client = genai.Client(api_key=api_key)
        
        # Test basic connection
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say 'Connected' in one word only"
        )
        
        # Get available image models
        image_models = []
        all_models = []
        
        for model in client.models.list():
            name = model.name if hasattr(model, 'name') else str(model)
            all_models.append(name)
            if 'image' in name.lower():
                image_models.append(name)
        
        # Test image generation capability
        image_status = test_image_generation(client)
        
        # Build status message
        status_parts = ["✅ API connection successful!"]
        
        if image_models:
            status_parts.append(f"\n**Available image models:**\n" + ", ".join(image_models[:5]))
        else:
            status_parts.append("\n⚠️ No image models found")
        
        status_parts.append(f"\n\n{image_status}")
        
        return True, "\n".join(status_parts)
        
    except Exception as e:
        error_msg = str(e)
        return False, f"❌ API connection failed:\n{error_msg}"


def test_image_generation(client) -> str:
    """Test if image generation works in user's region."""
    test_models = [
        "gemini-2.5-flash-preview-image-generation",
        "gemini-2.0-flash-exp",
    ]
    
    for model in test_models:
        try:
            # Try a simple image generation
            response = client.models.generate_content(
                model=model,
                contents="Generate a simple red circle",
                config=types.GenerateContentConfig(
                    response_modalities=["image", "text"],
                )
            )
            
            # Check if image was generated
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        return "✅ **Image generation:** Available in your region!"
            
        except Exception as e:
            error_msg = str(e).lower()
            if "not available in your country" in error_msg:
                return """
⚠️ **Image generation:** NOT available in your region

**Your options:**
1. Use VPN (connect to US/UK/EU)
2. Use alternative APIs (see below)
"""
            continue
    
    return "⚠️ **Image generation:** Status unknown"


def get_region_info() -> dict:
    """Try to get user's region info (for debugging)."""
    try:
        import urllib.request
        import json
        
        # Free IP geolocation API
        url = "https://ipapi.co/json/"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            return {
                "country": data.get("country_name", "Unknown"),
                "country_code": data.get("country_code", "??"),
                "region": data.get("region", "Unknown"),
                "city": data.get("city", "Unknown"),
            }
    except:
        return {"country": "Unknown", "country_code": "??"}