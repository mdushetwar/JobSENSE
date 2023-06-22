
#Libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from spacy import displacy
import nltk
import re
import string
import spacy
from num2words import num2words
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import plotly.graph_objects as go
from keybert import KeyBERT
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import random
import streamlit.components.v1 as com

# "-----------------------Backend Model------------------------------"

#Requirements
nltk.download('stopwords')
nltk.download('punkt')

stop_words=stopwords.words('english')
stemmer=SnowballStemmer('english')

def text_processor(text, lower=True, stop_word=True, sym_replace=True, short_form_replace=True, remove_punctuation=True, num_to_word=True, stem=True):

  if lower==True:
    #lower casing
    text=text.lower()

  #removing stopwords
  if stop_word==True:
    text_list=text.split()

    new_text=[]

    for word in text_list:
      if word not in stop_words:
        new_text.append(word)

    text=' '.join(new_text)

  #replacing short forms and symobols
  if sym_replace==True:
    text=text.replace('\n', '')
    text=text.replace('=', ' is equal to ')
    text=text.replace('%', ' percent ')
    text = text.replace('$', ' dollar ')
    text = text.replace('₹', ' rupee ')
    text = text.replace('€', ' euro ')
    text = text.replace('@', ' at ')
    text = text.replace('+', ' plus ')
    text = text.replace('/', ' or ')
    text = text.replace('-', ' ')
    text = text.replace('[math]', '')

  #replacement dict
  if short_form_replace==True:
    replacements_dict = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
        "power bi": "PowerBI",
        "Power BI": "PowerBI",
    }

    # Apply replacements
    pattern = re.compile(r'\b(' + '|'.join(replacements_dict.keys()) + r')\b')
    text = pattern.sub(lambda x: replacements_dict[x.group()], text)

  # Remove punctuation
  if remove_punctuation==True:
    text = text.translate(str.maketrans("", "", string.punctuation))

  #converting numbers to words
  if num_to_word==True:
    text_list=text.split()

    converted_text=[]

    for word in text_list:
      if word.isdigit():
        try:
          converted_word = num2words(float(word))
          converted_text.append(converted_word)
        except ValueError:
          converted_text.append(word)

      else:
        converted_text.append(word)

    text=' '.join(converted_text)

    #stemming
    if stem==True:
      text_stem=[stemmer.stem(word.strip()).strip() for word in text.split()]
      text=' '.join(text_stem)

  return text

#loading ner_model
ner_model=pickle.load(open('ner_model.pkl', 'rb'))

#keyBERT model
keyword_model=KeyBERT()

def get_keywords(text, num=10, ngram_range=(1,1), maxsum=False, mmr=False, candidate=10, diverse=0.1):
    text_new=text_processor(text, lower=True, stop_word=True, remove_punctuation=False, stem=False)
    keywords=keyword_model.extract_keywords(text_new, keyphrase_ngram_range=(1, 1), top_n=num, use_maxsum=maxsum, use_mmr=mmr, nr_candidates=candidate, diversity=diverse)
    keywords_dict={}
    for word, score in keywords:
        keywords_dict[word]=score

    return keywords_dict

def get_entities(text, ent_name):
  entities=[]
  doc=ner_model(text)
  for ent in doc.ents:
    if ent.label_==ent_name:
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


def visualize_cloud(text, font):

  new_text=text_processor(text,stem=False)
  keywords=get_keywords(new_text, 30)
  words=list(keywords.keys())
  random.seed(42)
  colors = [random.choice(['#FF4081', '#536DFE', '#00BCD4', '#FF9800', '#9C27B0', '#FF5722']) for _ in range(len(words))]

  wordcloud = WordCloud(
      background_color='white',
      width=1000,
      height=600,
      colormap='Paired',
      prefer_horizontal=0.9,
      font_path=font
  )

  wordcloud.generate_from_frequencies(dict(zip(words, [1]*len(words))))


  wordcloud.recolor(color_func=lambda *args, **kwargs: random.choice(colors))

  fig, ax=plt.subplots(figsize=(15, 10))
  ax.imshow(wordcloud, interpolation='bilinear')
  ax.axis('off')
  st.pyplot(fig)


def entity_score(text, entity):
  doc=ner_model(text)
  ent_list=[]
  for ent in doc.ents:
    if ent.label_==entity:
      text1=ent.text
      text1=text1.lower()
      ent_list.append(text1)
    
  ent_corpus=[]
  for ent1 in ent_list:
    for a in ent1.split():
      ent_corpus.append(a)

  keyword_dict=get_keywords(text, 100)

  ent_corp_dict={}
  for e2 in ent_corpus:
    if e2 in keyword_dict.keys():
      ent_corp_dict[e2]=keyword_dict[e2]
    else:
      ent_corp_dict[e2]=0

  ent_score={}
  for ent2 in ent_list:
    score=0
    for w in ent2.split():
      score=score+ent_corp_dict[w]
    avg_score=score/len(ent2.split())
    ent_score[ent2]=avg_score

  return ent_score

# "---------------------------Streamlit App----------------------------"

#sidebar
st.sidebar.markdown("# JobSENSE")

text=st.sidebar.text_area("Job Description", height=300)
rad=st.sidebar.radio('What would you like to see:', options=['TextView', 'Graphs', 'WordCloud'])
main_button=st.sidebar.button("Decode this Job")

st.sidebar.markdown("""
---
Created by 💥 [Mayur Dushetwar](https://www.mayurdushetwar.com) 
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

    st.write("")
    st.write("")
    st.markdown("""
    ##### A Beginner's Tutorial
    """)
    st.write("")
    st.write("")

    st.video("tutorial.mp4")

if main_button:
    if text.strip() != "":
        if rad=="TextView":

            st.markdown("# Overview")
            visualize_text_view(text)

            st.write("### Job Components")
            st.write("")
            tech=pd.DataFrame(get_entities(text, 'TECH'), columns=["TECH SKILLS"])
            soft=pd.DataFrame(get_entities(text, 'SOFT'), columns=["SOFT SKILLS"])
            tool=pd.DataFrame(get_entities(text, 'TOOL'), columns=["TOOLS/MODELS/FRAMEWORKS"])
            edu=pd.DataFrame(get_entities(text, 'EDU'), columns=["EDUCATION/DEGREE"])
            exp=pd.DataFrame(get_entities(text, 'EXP'), columns=["WORK EXPERIENCE"])
            domain=pd.DataFrame(get_entities(text, 'DOM'), columns=["DOMAIN"])

            col1, col2, col3=st.columns(3)
            col1.dataframe(tech, hide_index=True, width=500)
            col2.dataframe(tool, hide_index=True, width=500)
            col3.dataframe(soft, hide_index=True, width=500)

            col4, col5, col6=st.columns(3)
            col4.dataframe(edu, hide_index=True, width=500)
            col5.dataframe(exp, hide_index=True, width=500)
            col6.dataframe(domain, hide_index=True, width=500)


        if rad=="Graphs":
            c1, c2=st.columns((5,5))

            #row1
            with c1:
                new_text = text_processor(text, stem=False)
                keywords_10 = get_keywords(new_text, 10)
                df = pd.DataFrame()
                df['word'] = keywords_10.keys()
                df['score'] = keywords_10.values()
                
                fig = go.Figure(go.Pie(
                                labels=df['word'],
                                values=df['score']
                            ))

                fig.update_layout(
                    title="OVERALL KEYWORD SCORES",
                    showlegend=False
                            )
                fig.update_traces(textinfo='label+percent')
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                ent_score_dict = entity_score(text, "TECH")
                df2 = pd.DataFrame()
                df2['word'] = ent_score_dict.keys()
                df2['score'] = ent_score_dict.values()
                
                fig2 = go.Figure(go.Pie(
                                labels=df2['word'],
                                values=df2['score']
                            ))

                fig2.update_layout(
                    title="TECH WORD SCORES",
                    showlegend=False
                            )
                fig2.update_traces(textinfo='label+percent')
                st.plotly_chart(fig2, use_container_width=True)

            #row2
            c3,c4=st.columns((5,5))
                
            with c3:
                ent_score_dict = entity_score(text, "SOFT")
                df3 = pd.DataFrame()
                df3['word'] = ent_score_dict.keys()
                df3['score'] = ent_score_dict.values()
                
                fig3 = go.Figure(go.Pie(
                                labels=df3['word'],
                                values=df3['score']
                            ))

                fig3.update_layout(
                    title="SOFT WORD SCORES",
                    showlegend=False
                            )
                fig3.update_traces(textinfo='label+percent')
                st.plotly_chart(fig3, use_container_width=True)

            with c4:
                ent_score_dict = entity_score(text, "TOOL")
                df4 = pd.DataFrame()
                df4['word'] = ent_score_dict.keys()
                df4['score'] = ent_score_dict.values()
                
                fig4 = go.Figure(go.Pie(
                                labels=df4['word'],
                                values=df4['score']
                            ))

                fig4.update_layout(
                    title="TOOL/FRAME WORD SCORES",
                    showlegend=False
                            )
                fig4.update_traces(textinfo='label+percent')
                st.plotly_chart(fig4, use_container_width=True)

        if rad=="WordCloud":
            
            font = "/Users/mayurdushetwar/Documents/GitHub/job_post_analyser/fonts/ChrustyRock-ORLA.ttf"
            visualize_cloud(text, font)
