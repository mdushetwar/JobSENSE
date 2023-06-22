# Libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from spacy import displacy
import streamlit.components.v1 as com

# "-----------------------Backend Model------------------------------"

# loading ner_model
ner_model = pickle.load(open('ner_model.pkl', 'rb'))

def get_entities(text, ent_name):
    entities = []
    doc = ner_model(text)
    for ent in doc.ents:
        if ent.label_ == ent_name:
            entities.append(ent.text.lower())
    return np.unique(entities)


def visualize_text_view(text):
    colors = {
        "TOOL": "#F4A6C1",
        "TECH": "#F791A9",
        "SOFT": "#F9C894",
        "ROLE": "#FDC2A0",
        "DEPT": "#FFD19D",
        "EXP": "#AADBC9",
        "EDU": "#D2EAA8",
        "DOM": "#FFC8B2",
        "LANG": "#FF8E9D",
        "CERT": "#FFA1B3",
        "LOC": "#FFCC9E"
    }

    options = {
        'colors': colors,
        'compact': True,
    }

    doc = ner_model(text)
    html = displacy.render(doc, style='ent', options=options)
    com.html(html, width=800, height=400, scrolling=True)


# "---------------------------Streamlit App----------------------------"

# sidebar
st.sidebar.markdown("# JobSENSE")

text = st.sidebar.text_area("Job Description", height=300)
main_button = st.sidebar.button("Decode this Job")

st.sidebar.markdown("""
---
Created by [Mayur Dushetwar](https://www.mayurdushetwar.com) 
""")

if not main_button:
    st.markdown("""

    # Welcome to JobSENSE!
    """)

    com.iframe("https://embed.lottiefiles.com/animation/146741")

    st.markdown("""
    ### Here are some tips to maximize your results and enhance your experience:

    ##### 1. Light mode is recommended for better viewing experience
    ##### 2. Break down lengthy job descriptions into manageable sections
    ##### 3. Better visual experience with concise, 10-15 line descriptions
    ##### 4. Enhance results by using one sentence per line
    ##### 5. Current update excels in delivering better results for data-related job positions
    ##### 6. While JobSENSE also caters to other job types, accuracy may slightly vary

    """)


if main_button:
    if text.strip() != "":
        st.markdown("# Overview")
        visualize_text_view(text)

        st.write("### Job Components")
        st.write("")
        tech = pd.DataFrame(get_entities(text, 'TECH'), columns=["TECH SKILLS"])
        soft = pd.DataFrame(get_entities(text, 'SOFT'), columns=["SOFT SKILLS"])
        tool = pd.DataFrame(get_entities(text, 'TOOL'), columns=["TOOLS/MODELS/FRAMEWORKS"])
        edu = pd.DataFrame(get_entities(text, 'EDU'), columns=["EDUCATION/DEGREE"])
        exp = pd.DataFrame(get_entities(text, 'EXP'), columns=["WORK EXPERIENCE"])
        domain = pd.DataFrame(get_entities(text, 'DOM'), columns=["DOMAIN"])

        col1, col2, col3 = st.columns(3)
        col1.dataframe(tech, hide_index=True, width=500)
        col2.dataframe(tool, hide_index=True, width=500)
        col3.dataframe(soft, hide_index=True, width=500)

        col4, col5, col6 = st.columns(3)
        col4.dataframe(edu, hide_index=True, width=500)
        col5.dataframe(exp, hide_index=True, width=500)
        col6.dataframe(domain, hide_index=True, width=500)
