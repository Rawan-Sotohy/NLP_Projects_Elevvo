"""
Streamlit Web App for Named Entity Recognition - FAST VERSION
Usage: streamlit run app.py
"""

import streamlit as st
import re
from collections import Counter
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="NER - Named Entity Recognition",
    page_icon="🏷️",
    layout="wide"
)


def simple_ner_extract(text):
    """
    Simple NER extraction using patterns (FAST - no model loading!)
    This is a demo version - replace with real model later
    """
    entities = []
    
    # Pattern for capitalized words (likely names/places)
    # This finds words that start with capital letters
    pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    
    matches = re.finditer(pattern, text)
    
    # Common patterns for different entity types
    person_keywords = ['CEO', 'President', 'Mr', 'Mrs', 'Dr', 'scored', 'met', 'announced']
    location_keywords = ['in', 'at', 'from', 'Stadium', 'City', 'Egypt', 'England', 'Paris', 'California']
    org_keywords = ['Inc', 'Corp', 'Company', 'United', 'University', 'Organization']
    
    for match in matches:
        word = match.group()
        start = match.start()
        end = match.end()
        
        # Get context around the word
        context_start = max(0, start - 30)
        context_end = min(len(text), end + 30)
        context = text[context_start:context_end].lower()
        
        # Determine entity type based on context
        entity_type = "MISC"
        confidence = 0.75
        
        # Check for person indicators
        if any(keyword.lower() in context for keyword in person_keywords):
            entity_type = "PER"
            confidence = 0.85
        # Check for location indicators
        elif any(keyword.lower() in context for keyword in location_keywords):
            entity_type = "LOC"
            confidence = 0.80
        # Check for organization indicators
        elif any(keyword.lower() in word for keyword in org_keywords) or 'Inc' in word:
            entity_type = "ORG"
            confidence = 0.88
        
        entities.append({
            'word': word,
            'entity_group': entity_type,
            'score': confidence,
            'start': start,
            'end': end
        })
    
    return entities


def get_entity_color(entity_type):
    """Get color for each entity type"""
    colors = {
        "PER": "#FF6B6B",
        "ORG": "#4ECDC4",
        "LOC": "#95E1D3",
        "MISC": "#F9CA24"
    }
    return colors.get(entity_type, "#95A5A6")


def highlight_entities(text, entities):
    """Create highlighted HTML for entities"""
    if not entities:
        return text
    
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    highlighted_text = text
    for entity in sorted_entities:
        start = entity['start']
        end = entity['end']
        word = entity['word']
        entity_type = entity['entity_group']
        color = get_entity_color(entity_type)
        
        replacement = f'<span style="background-color: {color}; padding: 2px 6px; border-radius: 3px; color: white; font-weight: bold;">{word} <sub style="font-size: 10px;">({entity_type})</sub></span>'
        
        highlighted_text = highlighted_text[:start] + replacement + highlighted_text[end:]
    
    return highlighted_text


def main():
    """Main Streamlit app"""
    
    # Title with gradient background
    st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 30px;">
            <h1 style="color: white; margin: 0;">🏷️ Named Entity Recognition System</h1>
            <p style="color: white; margin: 10px 0 0 0; font-size: 18px;">Extract entities (Person, Organization, Location, Miscellaneous) from text</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.5, 0.05
    )
    
    st.sidebar.markdown("---")
    st.sidebar.success("✅ System Ready! (Fast Demo Mode)")
    st.sidebar.info("💡 Using pattern-based extraction for instant results")
    
    # Main content
    st.markdown("---")
    
    # Example texts
    example_texts = {
        "Example 1 - Tech": "Apple Inc. CEO Tim Cook announced new products at an event in Cupertino, California.",
        "Example 2 - Sports": "Cristiano Ronaldo scored two goals for Manchester United against Liverpool at Old Trafford Stadium in England.",
        "Example 3 - Politics": "President Joe Biden met with Emmanuel Macron in Paris to discuss NATO policies."
    }
    
    # Input selection
    col1, col2 = st.columns([1, 1])
    with col1:
        use_example = st.checkbox("📋 Use Example Text", value=False)
    
    if use_example:
        with col2:
            selected_example = st.selectbox("Choose example:", list(example_texts.keys()))
        default_text = example_texts[selected_example]
    else:
        default_text = ""
    
    st.markdown("---")
    
    # Main text area
    st.markdown("### ✍️ ENTER YOUR TEXT HERE:")
    
    text_input = st.text_area(
        label="Text to analyze",
        value=default_text,
        height=250,
        placeholder="Type or paste your text here...\n\nExample:\nApple Inc. CEO Tim Cook visited Cairo, Egypt and announced new partnerships with local companies.\n\nOr try: Mohamed Salah plays for Liverpool Football Club in England.",
        label_visibility="collapsed",
        key="main_text_area"
    )
    
    st.markdown("---")
    
    # Analyze button
    analyze_button = st.button(
        "🔍 ANALYZE TEXT & EXTRACT ENTITIES",
        type="primary",
        use_container_width=True,
        key="analyze_btn"
    )
    
    # Results section
    if analyze_button:
        if text_input.strip():
            # Fast extraction - no loading time!
            entities = simple_ner_extract(text_input)
            
            filtered_entities = [
                e for e in entities 
                if e['score'] >= confidence_threshold
            ]
            
            if filtered_entities:
                st.success(f"✅ Successfully extracted {len(filtered_entities)} entities!")
                
                st.markdown("---")
                
                # Highlighted text
                st.markdown("### 📝 Annotated Text")
                highlighted = highlight_entities(text_input, filtered_entities)
                st.markdown(
                    f'<div style="background-color: #ffffff; padding: 25px; border-radius: 10px; border: 2px solid #e0e0e0; line-height: 2.2; font-size: 16px;">{highlighted}</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown("---")
                
                # Results in two columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 📋 Extracted Entities Table")
                    
                    df = pd.DataFrame([
                        {
                            "Entity": e['word'],
                            "Type": e['entity_group'],
                            "Confidence": f"{e['score']:.1%}"
                        }
                        for e in filtered_entities
                    ])
                    
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )
                
                with col2:
                    st.markdown("### 📊 Statistics")
                    
                    entity_counts = Counter([e['entity_group'] for e in filtered_entities])
                    
                    for entity_type, count in entity_counts.most_common():
                        color = get_entity_color(entity_type)
                        st.markdown(
                            f'<div style="background-color: {color}; padding: 15px; margin: 10px 0; border-radius: 8px; color: white; font-weight: bold; text-align: center; font-size: 18px;">'
                            f'{entity_type}: {count}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    st.metric("Total Entities", len(filtered_entities))
                    
                    avg_confidence = sum(e['score'] for e in filtered_entities) / len(filtered_entities)
                    st.metric("Average Confidence", f"{avg_confidence:.1%}")
            
            else:
                st.warning("⚠️ No entities found! Try:\n- Lowering the confidence threshold\n- Using capitalized names (e.g., 'John Smith' not 'john smith')")
        else:
            st.error("❌ Please enter some text first!")
    
    # Entity types reference
    st.markdown("---")
    st.markdown("### 📖 Entity Types Reference")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            '<div style="background-color: #FF6B6B; padding: 20px; border-radius: 10px; color: white; text-align: center; min-height: 150px;">'
            '<h3 style="margin: 0; color: white;">👤 PERSON</h3>'
            '<p style="margin: 10px 0;">Names of people</p>'
            '<small><i>Tim Cook, Mohamed Salah</i></small>'
            '</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            '<div style="background-color: #4ECDC4; padding: 20px; border-radius: 10px; color: white; text-align: center; min-height: 150px;">'
            '<h3 style="margin: 0; color: white;">🏢 ORGANIZATION</h3>'
            '<p style="margin: 10px 0;">Companies, agencies</p>'
            '<small><i>Apple Inc., NASA</i></small>'
            '</div>',
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            '<div style="background-color: #95E1D3; padding: 20px; border-radius: 10px; color: white; text-align: center; min-height: 150px;">'
            '<h3 style="margin: 0; color: white;">📍 LOCATION</h3>'
            '<p style="margin: 10px 0;">Places, countries</p>'
            '<small><i>California, Cairo</i></small>'
            '</div>',
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            '<div style="background-color: #F9CA24; padding: 20px; border-radius: 10px; color: white; text-align: center; min-height: 150px;">'
            '<h3 style="margin: 0; color: white;">🏷️ MISCELLANEOUS</h3>'
            '<p style="margin: 10px 0;">Events, products</p>'
            '<small><i>iPhone, World Cup</i></small>'
            '</div>',
            unsafe_allow_html=True
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #999; padding: 20px;">'
        '<p>Built with Streamlit | ⚡ Fast Demo Mode (Pattern-Based Extraction)</p>'
        '<p style="font-size: 12px; margin-top: 10px;">Note: This uses simple pattern matching for instant results. For production, replace with trained NER model.</p>'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()