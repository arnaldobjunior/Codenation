#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[108]:


import pandas as pd
import numpy as np


# In[109]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[110]:



black_friday.head()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[140]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[141]:


def q2():
    b = black_friday.loc[(black_friday["Gender"] == "F") & (black_friday["Age"] == "26-35")].shape[0]
    # Retorne aqui o resultado da questão 2.
    return b


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[142]:


def q3():
    c=len(black_friday["User_ID"].unique())
    # Retorne aqui o resultado da questão 3.
    return c


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[143]:


def q4():
    d=black_friday.duplicated(subset='User_ID', keep='first').sum()
    # Retorne aqui o resultado da questão 4.
    return np.int(d)


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[144]:


def q5():
    none = pd.DataFrame(pd.isnull(black_friday).sum(axis=1))
    porc = none[none[0]!=0].count()[0]
    e= porc/black_friday.shape[0]
    # Retorne aqui o resultado da questão 5.
    return np.float(e)


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[145]:


def q6():
    f=max((pd.isnull(black_friday).sum(axis=0)).values)
    # Retorne aqui o resultado da questão 6.
    return np.int(f)


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[146]:


def q7():
    valores = black_friday["Product_Category_3"].value_counts().index.tolist()
    
    # Retorne aqui o resultado da questão 7.
    return valores[0]


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[151]:


def q8():
    from sklearn import preprocessing

    purchase =  pd.DataFrame(black_friday["Purchase"])

    scaler = preprocessing.MinMaxScaler()
    scaler_fit = scaler.fit_transform(purchase)
    purchase_norm = pd.DataFrame(scaler_fit)
    purchase_mean = purchase_norm.mean()
    
    # Retorne aqui o resultado da questão 8.
    return np.float(purchase_mean[0])


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[150]:


def q9():
    from sklearn import preprocessing

    purchase =  pd.DataFrame(black_friday["Purchase"])

    scaler = preprocessing.MinMaxScaler()
    scaler_fit = scaler.fit_transform(purchase)
    purchase_norm = pd.DataFrame(scaler_fit)
    g=purchase_norm[ (purchase_norm[0] ==1)| (purchase_norm[0]== -1) ].count()[0]
    # Retorne aqui o resultado da questão 9.
    return np.int(g)


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[149]:


def q10(): 
    cat_boolean = black_friday[["Product_Category_2","Product_Category_3"]]
    cat_boolean = pd.isnull(cat_boolean)
    h=not(any((cat_boolean["Product_Category_2"]== True) & (cat_boolean["Product_Category_3"]== True)== False))
    # Retorne aqui o resultado da questão 10.
    return h 


# In[ ]:





# In[ ]:




