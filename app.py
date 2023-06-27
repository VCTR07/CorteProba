import streamlit as st


import csv
import pandas as pd
from collections import Counter
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
#from imblearn.under_sampling import NearMiss
import nltk
nltk.download('stopwords')
stopwords=nltk.corpus.stopwords.words("portuguese")
nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer() #ESSE STEMMER É FEITO PARA A LINGUA PORTUGUESA
nltk.download("punkt")
from itertools import islice
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
#from goose3 import Goose
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
nlp = spacy.load('pt_core_news_sm')

resultados=pd.read_csv('resultados.csv')

def sentence_tokenizer(sentence):
    return [token.lemma_ for token in nlp(sentence.lower()) if (token.is_alpha & ~token.is_stop)]

def normalizer(sentence):
    tokenized_sentence = sentence_tokenizer(sentence)
    return ' '.join(tokenized_sentence)

#Histograma
def plot_hist(df):
    fig = go.Figure(data=[go.Histogram(x=df['resultados'])])
    #fig = go.Figure([go.Bar(x=df.Palavras, y=df.Quantidade, text=df.Quantidade, textposition='auto')])
    fig.update_layout(
        autosize=False,
        width=500,
        height=500)
    return fig  

#load modelo NB
filename='classif_s_f_crosVal.sav'
loaded_model = pickle.load(open(filename, 'rb'))
#load modelo tidf
filename1='tfidf_s_f.sav'
loaded_model1 = pickle.load(open(filename1, 'rb'))

st.title("Insira um acórdão recorrido para verificar a possibilidade de sucesso ou fracasso do recurso")

with st.form(key='includ_avaliacao'):    
    input_avaliacao = st.text_input(label = "Acórdão recorrido")    
    input_button_submit = st.form_submit_button("Enviar")


if input_button_submit:
    resultado1=None
    resultado1=loaded_model.predict(loaded_model1.transform([normalizer(input_avaliacao)]))
    if resultado1 is not None:
        st.write(f'Se retornou 0 indica fracasso e se retornou 1 indica sucesso: {resultado1[0]}')
        novaEntrada=resultado1[0]
        resultados.loc[len(resultados)] = novaEntrada
        #with open('resultados.csv', 'w', newline='') as csvfile:
            #writer = csv.writer(csvfile, dialect='unix')
        st.subheader('Histograma de resultados')
        st.plotly_chart(plot_hist(resultados))




# st.title("Avalie nossa Empresa")

# with st.form(key='includ_avaliacao'):
#     input_correcao=None
#     input_avaliacao = st.text_input(label = "Digite sua avaliação sobre o produto ou serviço")
#     if input_avaliacao:
#         resultado=nb.predict(tfidf_vecorizer.transform([normalizer(input_avaliacao)]))
#         st.write(f'Classificação automática da avaliação: {resultado}')
#     input_correcao= st.selectbox("Se a classificação da sua avaliação não está correta, por favor reclassifique selecionando uma das opções abaixo", ["Excelente", "Bom", "Regular", "Ruim", "Péssimo"])
#     input_button_submit = st.form_submit_button("Enviar")

# if input_button_submit:
#     st.write(f'Sua Avaliação: {input_avaliacao}')
#     if input_correcao is None:
#         st.write(f'Classificação automática: {resultado}')        
#     else:
#         st.write(f'Classificação manual: {input_correcao}')

