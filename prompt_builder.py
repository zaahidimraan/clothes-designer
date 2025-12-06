"""
Builds optimized prompts for Gemini image generation based on user selections.
"""

def build_design_prompt(selections: dict) -> str:
    """
    Convert user selections into a detailed prompt for image generation.
    """
    prompt_parts = []
    
    # Main subject
    clothing_type = selections.get("clothing_type", "dress")
    prompt_parts.append(f"A beautiful women's {clothing_type}")
    
    # Fabric
    if selections.get("fabric"):
        prompt_parts.append(f"made of {selections['fabric']} fabric")
    
    # Color
    if selections.get("base_color"):
        prompt_parts.append(f"in {selections['base_color']} color")
    
    # Pattern
    if selections.get("pattern") and selections["pattern"] != "Solid/Plain":
        prompt_parts.append(f"with {selections['pattern']} pattern")
    
    # Neckline
    if selections.get("neckline"):
        prompt_parts.append(f"featuring a {selections['neckline']}")
    
    # Sleeves
    if selections.get("sleeve_style"):
        prompt_parts.append(f"with {selections['sleeve_style']}")
    
    # Fit
    if selections.get("fit_style"):
        prompt_parts.append(f"{selections['fit_style']}")
    
    # Buttons
    if selections.get("button_style") and selections["button_style"] != "No Buttons":
        prompt_parts.append(f"with {selections['button_style']}")
    
    # Embellishments
    if selections.get("embellishments") and selections["embellishments"] != "None":
        prompt_parts.append(f"decorated with {selections['embellishments']}")
    
    # Custom description
    if selections.get("custom_description"):
        prompt_parts.append(f". Additional details: {selections['custom_description']}")
    
    base_prompt = " ".join(prompt_parts)
    
    full_prompt = f"""Generate a high-quality fashion design image:

{base_prompt}

Style requirements:
- Professional fashion illustration or realistic product photo style
- Clean white or light gradient background
- Show the full garment clearly
- High detail on fabric texture and embellishments
- Elegant and fashionable presentation
- No model/person, just the clothing item displayed beautifully"""
    
    return full_prompt


def build_modification_prompt(original_prompt: str, modification_request: str) -> str:
    """
    Build a prompt for modifying an existing design.
    """
    return f"""Modify the following clothing design based on the requested changes:

ORIGINAL DESIGN:
{original_prompt}

REQUESTED MODIFICATIONS:
{modification_request}

Generate a new design incorporating these changes while maintaining quality.

Style requirements:
- Professional fashion illustration style
- Clean white or light gradient background
- Show the full garment clearly
- High detail on fabric texture
- No model/person, just the clothing item"""


def get_prompt_summary(selections: dict) -> str:
    """
    Generate a human-readable summary of the design selections.
    """
    summary_parts = []
    
    labels = {
        'clothing_type': 'Type',
        'base_color': 'Color',
        'fabric': 'Fabric',
        'pattern': 'Pattern',
        'neckline': 'Neckline',
        'sleeve_style': 'Sleeves',
        'fit_style': 'Fit',
        'button_style': 'Buttons',
        'embellishments': 'Embellishments',
        'custom_description': 'Custom Details'
    }
    
    for key, label in labels.items():
        if selections.get(key):
            summary_parts.append(f"**{label}:** {selections[key]}")
    
    return "\n".join(summary_parts)