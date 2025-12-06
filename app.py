"""
Women's Clothing Designer - Streamlit Application
Design custom clothes using AI-powered image generation.
"""

import streamlit as st
from io import BytesIO

# Import local modules
from config import (
    CLOTHING_TYPES, COLORS, PATTERNS, NECKLINES,
    SLEEVE_STYLES, BUTTON_STYLES, EMBELLISHMENTS,
    FABRICS, FIT_STYLES
)
from prompt_builder import build_design_prompt, get_prompt_summary, build_modification_prompt
from gemini_service import GeminiDesignService, test_api_connection, GENAI_AVAILABLE

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="AI Clothes Designer",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== CUSTOM CSS ==============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E91E63;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #333;
        border-bottom: 2px solid #E91E63;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .design-summary {
        background-color: #fce4ec;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #E91E63;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)


# ============== SESSION STATE ==============
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'generated_image': None,
        'current_prompt': "",
        'design_history': [],
        'api_key': "",
        'selections': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============== SIDEBAR ==============
def render_sidebar():
    """Render sidebar with configuration."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # API Key
        st.markdown("### 🔑 API Key")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.api_key,
            help="Get your key from: https://aistudio.google.com/app/apikey",
            label_visibility="collapsed",
            placeholder="Enter your Gemini API key..."
        )
        
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        # Test button
        if st.button("🔗 Test Connection", use_container_width=True):
            if api_key:
                with st.spinner("Testing connection..."):
                    success, message = test_api_connection(api_key)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("Please enter an API key first.")
        
        st.markdown("---")
        
        # Instructions
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Enter your Gemini API key
        2. Select clothing options
        3. Add custom details (optional)
        4. Click **Generate Design**
        5. Modify if needed!
        """)
        
        st.markdown("---")
        
        # Design History
        st.markdown("### 📜 Design History")
        if st.session_state.design_history:
            for i, design in enumerate(reversed(st.session_state.design_history[-5:])):
                idx = len(st.session_state.design_history) - i
                with st.expander(f"Design #{idx}"):
                    sel = design.get('selections', {})
                    st.write(f"**Type:** {sel.get('clothing_type', 'N/A')}")
                    st.write(f"**Color:** {sel.get('base_color', 'N/A')}")
                    st.write(f"**Pattern:** {sel.get('pattern', 'N/A')}")
        else:
            st.info("No designs yet.")
        
        st.markdown("---")
        
        # Clear button
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.generated_image = None
            st.session_state.current_prompt = ""
            st.session_state.selections = {}
            st.rerun()


# ============== DESIGN FORM ==============
def render_design_form():
    """Render the design options form."""
    selections = {}
    
    # ===== BASIC OPTIONS =====
    st.markdown('<p class="section-header">👗 Basic Options</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selections['clothing_type'] = st.selectbox(
            "Clothing Type *",
            options=CLOTHING_TYPES,
            index=0
        )
    
    with col2:
        color_option = st.selectbox(
            "Base Color *",
            options=["Select from list", "Custom color"]
        )
        if color_option == "Select from list":
            selections['base_color'] = st.selectbox(
                "Choose color",
                options=COLORS,
                label_visibility="collapsed"
            )
        else:
            selections['base_color'] = st.text_input(
                "Enter color",
                placeholder="e.g., Pastel mint green",
                label_visibility="collapsed"
            )
    
    with col3:
        selections['fabric'] = st.selectbox(
            "Fabric Type",
            options=FABRICS,
            index=0
        )
    
    # ===== STYLE OPTIONS =====
    st.markdown('<p class="section-header">✨ Style Options</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pattern_option = st.selectbox(
            "Pattern",
            options=["Select from list", "Custom pattern"]
        )
        if pattern_option == "Select from list":
            selections['pattern'] = st.selectbox(
                "Choose pattern",
                options=PATTERNS,
                label_visibility="collapsed"
            )
        else:
            selections['pattern'] = st.text_input(
                "Describe pattern",
                placeholder="e.g., Small roses with leaves",
                label_visibility="collapsed"
            )
    
    with col2:
        selections['neckline'] = st.selectbox(
            "Neckline",
            options=NECKLINES,
            index=0
        )
    
    with col3:
        selections['sleeve_style'] = st.selectbox(
            "Sleeve Style",
            options=SLEEVE_STYLES,
            index=2
        )
    
    # ===== FIT & DETAILS =====
    st.markdown('<p class="section-header">🎨 Fit & Details</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selections['fit_style'] = st.selectbox(
            "Fit Style",
            options=FIT_STYLES,
            index=0
        )
    
    with col2:
        selections['button_style'] = st.selectbox(
            "Button Style",
            options=BUTTON_STYLES,
            index=0
        )
    
    with col3:
        selections['embellishments'] = st.selectbox(
            "Embellishments",
            options=EMBELLISHMENTS,
            index=0
        )
    
    # ===== CUSTOM DESCRIPTION =====
    st.markdown('<p class="section-header">✏️ Custom Details (Optional)</p>', unsafe_allow_html=True)
    
    selections['custom_description'] = st.text_area(
        "Add any additional details for your design",
        placeholder="Example: Add a small bow at the waist, make the collar slightly oversized, add subtle gold threading, vintage style buttons...",
        height=100,
        label_visibility="collapsed"
    )
    
    return selections


# ============== DESIGN SUMMARY ==============
def render_design_summary(selections):
    """Show summary of selected options."""
    st.markdown('<div class="design-summary">', unsafe_allow_html=True)
    st.markdown("#### 📋 Your Design Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Type:** {selections.get('clothing_type', 'N/A')}")
        st.markdown(f"**Color:** {selections.get('base_color', 'N/A')}")
        st.markdown(f"**Fabric:** {selections.get('fabric', 'N/A')}")
        st.markdown(f"**Pattern:** {selections.get('pattern', 'N/A')}")
        st.markdown(f"**Embellishments:** {selections.get('embellishments', 'N/A')}")
    
    with col2:
        st.markdown(f"**Neckline:** {selections.get('neckline', 'N/A')}")
        st.markdown(f"**Sleeves:** {selections.get('sleeve_style', 'N/A')}")
        st.markdown(f"**Fit:** {selections.get('fit_style', 'N/A')}")
        st.markdown(f"**Buttons:** {selections.get('button_style', 'N/A')}")
    
    if selections.get('custom_description'):
        st.markdown(f"**Custom Details:** {selections['custom_description']}")
    
    st.markdown('</div>', unsafe_allow_html=True)


# ============== GENERATE DESIGN ==============
def generate_design(selections, api_key):
    """Generate design using Gemini API."""
    if not api_key:
        st.error("⚠️ Please enter your Gemini API key in the sidebar.")
        return False
    
    if not selections.get('base_color'):
        st.error("⚠️ Please select or enter a base color.")
        return False
    
    # Build prompt
    prompt = build_design_prompt(selections)
    st.session_state.current_prompt = prompt
    st.session_state.selections = selections
    
    # Show prompt (collapsed)
    with st.expander("🔍 View Generated Prompt"):
        st.code(prompt, language="text")
    
    # Generate image
    with st.spinner("🎨 Creating your design... This may take 30-60 seconds..."):
        try:
            service = GeminiDesignService(api_key)
            image, message = service.generate_design(prompt)
            
            if image:
                st.session_state.generated_image = image
                st.session_state.design_history.append({
                    'selections': selections.copy(),
                    'prompt': prompt
                })
                st.success(f"✅ {message}")
                return True
            else:
                st.error(f"❌ {message}")
                return False
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return False


# ============== MODIFY DESIGN ==============
def modify_design(modification_text, api_key):
    """Modify existing design based on user input."""
    if not api_key:
        st.error("⚠️ Please enter your Gemini API key in the sidebar.")
        return False
    
    if not modification_text:
        st.error("⚠️ Please describe what you want to change.")
        return False
    
    if not st.session_state.current_prompt:
        st.error("⚠️ No existing design to modify. Generate a design first.")
        return False
    
    # Build modification prompt
    prompt = build_modification_prompt(st.session_state.current_prompt, modification_text)
    
    # Show prompt (collapsed)
    with st.expander("🔍 View Modification Prompt"):
        st.code(prompt, language="text")
    
    # Generate modified image
    with st.spinner("🎨 Modifying your design... This may take 30-60 seconds..."):
        try:
            service = GeminiDesignService(api_key)
            image, message = service.generate_design(prompt)
            
            if image:
                st.session_state.generated_image = image
                st.session_state.current_prompt = prompt
                st.session_state.design_history.append({
                    'selections': st.session_state.selections.copy(),
                    'prompt': prompt,
                    'modification': modification_text
                })
                st.success(f"✅ {message}")
                return True
            else:
                st.error(f"❌ {message}")
                return False
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return False


# ============== RESULT SECTION ==============
def render_result_section():
    """Render the generated image and modification options."""
    if st.session_state.generated_image is not None:
        st.markdown("---")
        st.markdown('<p class="section-header">🖼️ Your Generated Design</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display image
            st.image(
                st.session_state.generated_image,
                caption="AI Generated Clothing Design",
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 💾 Download")
            
            # Convert image to bytes for download
            img_buffer = BytesIO()
            st.session_state.generated_image.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Design (PNG)",
                data=img_bytes,
                file_name="clothing_design.png",
                mime="image/png",
                use_container_width=True
            )
            
            st.markdown("---")
            
            # Modification section
            st.markdown("#### ✏️ Modify Design")
            
            modification_text = st.text_area(
                "Describe changes you want:",
                placeholder="Example:\n- Change color to blue\n- Add lace on sleeves\n- Make it longer\n- Add floral embroidery",
                height=150,
                key="modification_input",
                label_visibility="collapsed"
            )
            
            if st.button("🔄 Apply Changes", use_container_width=True):
                modify_design(modification_text, st.session_state.api_key)
                st.rerun()
            
            st.markdown("---")
            
            # Quick modifications
            st.markdown("#### ⚡ Quick Changes")
            
            quick_col1, quick_col2 = st.columns(2)
            
            with quick_col1:
                if st.button("🎨 Change Color", use_container_width=True, key="quick_color"):
                    st.session_state.quick_mod = "color"
                
                if st.button("📏 Change Length", use_container_width=True, key="quick_length"):
                    st.session_state.quick_mod = "length"
            
            with quick_col2:
                if st.button("✨ Add Details", use_container_width=True, key="quick_details"):
                    st.session_state.quick_mod = "details"
                
                if st.button("🔄 Regenerate", use_container_width=True, key="quick_regen"):
                    generate_design(st.session_state.selections, st.session_state.api_key)
                    st.rerun()
            
            # Handle quick modifications
            if 'quick_mod' in st.session_state:
                mod_type = st.session_state.quick_mod
                
                if mod_type == "color":
                    new_color = st.selectbox("Select new color:", COLORS, key="new_color_select")
                    if st.button("Apply Color", key="apply_color"):
                        modify_design(f"Change the main color to {new_color}", st.session_state.api_key)
                        del st.session_state.quick_mod
                        st.rerun()
                
                elif mod_type == "length":
                    length_change = st.selectbox(
                        "Select length:",
                        ["Shorter/Crop", "Knee length", "Midi length", "Maxi/Long"],
                        key="length_select"
                    )
                    if st.button("Apply Length", key="apply_length"):
                        modify_design(f"Change the length to {length_change}", st.session_state.api_key)
                        del st.session_state.quick_mod
                        st.rerun()
                
                elif mod_type == "details":
                    detail_options = st.multiselect(
                        "Add details:",
                        ["Lace trim", "Embroidery", "Ruffles", "Sequins", "Bow", "Belt"],
                        key="detail_select"
                    )
                    if st.button("Apply Details", key="apply_details") and detail_options:
                        modify_design(f"Add these details: {', '.join(detail_options)}", st.session_state.api_key)
                        del st.session_state.quick_mod
                        st.rerun()


# ============== MAIN APP ==============
def main():
    """Main application entry point."""
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # ===== HEADER =====
    st.markdown('<p class="main-header">👗 AI Clothes Designer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Design beautiful custom clothing with AI-powered image generation</p>', unsafe_allow_html=True)
    
    # ===== CHECK DEPENDENCIES =====
    if not GENAI_AVAILABLE:
        st.error("""
        ⚠️ **Missing Dependency**
        
        The `google-genai` package is not installed. Please run:
        ```
        pip install google-genai
        ```
        """)
        st.stop()
    
    # ===== API KEY CHECK =====
    if not st.session_state.api_key:
        st.warning("""
        👋 **Welcome!** 
        
        To get started, please enter your **Gemini API key** in the sidebar.
        
        Don't have one? Get it free from [Google AI Studio](https://aistudio.google.com/app/apikey)
        """)
    
    # ===== DESIGN FORM =====
    st.markdown("---")
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["🎨 Design New", "📜 View Prompt"])
    
    with tab1:
        # Render form and get selections
        selections = render_design_form()
        
        st.markdown("---")
        
        # Show summary
        render_design_summary(selections)
        
        st.markdown("---")
        
        # Generate button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_clicked = st.button(
                "✨ Generate Design",
                type="primary",
                use_container_width=True
            )
            
            if generate_clicked:
                success = generate_design(selections, st.session_state.api_key)
                if success:
                    st.rerun()
    
    with tab2:
        if st.session_state.current_prompt:
            st.markdown("### Current Design Prompt")
            st.code(st.session_state.current_prompt, language="text")
            
            # Copy button
            st.download_button(
                "📋 Download Prompt",
                data=st.session_state.current_prompt,
                file_name="design_prompt.txt",
                mime="text/plain"
            )
        else:
            st.info("Generate a design to see the prompt here.")
    
    # ===== RESULT SECTION =====
    render_result_section()
    
    # ===== FOOTER =====
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.9rem;">
            Made with ❤️ using Streamlit & Google Gemini AI<br>
            <small>Note: Generated designs are AI-created and may vary in accuracy</small>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============== RUN APP ==============
if __name__ == "__main__":
    main()