#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[80]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


# In[82]:


countries = pd.read_csv("countries.csv")


# In[83]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[84]:


# Sua análise começa aqui.
countries= countries.replace(",",".",regex=True)


# In[85]:


countries.dtypes


# In[86]:


countries.iloc[:,3:21]= countries.iloc[:,3:21].astype(float)


# In[87]:


countries.head()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[88]:


def q1():
    region = countries["Region"].str.strip().unique()
    region = np.sort(region)
    return region.tolist()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[89]:


from sklearn.preprocessing import KBinsDiscretizer


# In[90]:


def q2():
    discretizar= KBinsDiscretizer(n_bins=10, encode="ordinal").fit_transform(countries[["Pop_density"]])
    paises90 = (discretizar>np.quantile(discretizar,0.9)).sum()
    return int(paises90)


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[91]:


from sklearn.preprocessing import OneHotEncoder


# In[92]:


nan = [countries["Region"].isnull().sum(),countries["Climate"].isnull().sum()]
nan


# In[93]:


def q3():
    encoder = OneHotEncoder(sparse=False, dtype=np.int)
    countries["Climate"] = countries[['Climate']].fillna(countries['Climate'].mean())
    climate_encoded = encoder.fit_transform(countries[["Climate"]])
    region_encoded = encoder.fit_transform(countries[["Region"]])
    return climate_encoded.shape[1]+region_encoded.shape[1]


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[94]:


countries.iloc[:,2:21].columns


# In[95]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[96]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# In[97]:


def q4():
    pipeline = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'median')),
                                 ('scaler', StandardScaler())])
    pipeline.fit(countries[countries.iloc[:,2:21].columns])
    pipeline = pipeline.transform([test_country[2:]])
    return float(pipeline[0][9].round(3))


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[108]:


def q5():
    q1 = countries["Net_migration"].quantile(0.25)
    q3 = countries["Net_migration"].quantile(0.75)
    iqr = q3-q1
    outliers_abaixo = (countries["Net_migration"] < (q1-1.5*iqr)).sum()
    outliers_acima = (countries["Net_migration"] > (q3+1.5*iqr)).sum()
    return (int(outliers_abaixo),int(outliers_acima),False)


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[99]:


from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import CountVectorizer
categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[100]:


def q6():
    vectorizer = CountVectorizer()
    newsgroup_fit = vectorizer.fit_transform(newsgroup['data'])
    count = vectorizer.get_feature_names().index('phone')
    return int(newsgroup_fit[:, count].sum())


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[101]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[102]:


def q7():
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(newsgroup['data'])
    newsgroups_tfidf_vectorized = tfidf_vectorizer.transform(newsgroup.data)
    count = tfidf_vectorizer.get_feature_names().index('phone')
    return float(newsgroups_tfidf_vectorized[:, count].sum().round(3))


# In[ ]:




