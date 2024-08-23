from operator import index
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Page configuration
st.set_page_config (
  page_title = 'Iris prediccion',
  layout = 'wide',
  initial_sidebar_state = 'expanded'
)

# Titulo de la app
st.title ('Prediccion clase de orquidea')
st.image("image/iris-machinelearning.jpg", caption="Flores")
# Lectura de datos
iris = load_iris ()
df = pd.DataFrame (iris.data, columns = iris.feature_names)

st.write (df)

st.sidebar.header("Entrada de datos")
sl=st.sidebar.slider("sepal length (cm)", 4.3,7.9,5.84)
sw=st.sidebar.slider("sepal width (cm)", 2.0, 4.4, 3.05)
pl=st.sidebar.slider("petal length (cm)", 1.0, 6.9, 3.76)
pw=st.sidebar.slider(" petal width (cm)", 0.1, 2.5, 1.20)

df ['target']=iris.target


X_train, X_test, y_train, y_test = train_test_split (
    df.drop (['target'], axis='columns'), iris.target,
    test_size = 0.2
)

model = RandomForestClassifier ()
model.fit (X_train, y_train)

st.write("sepal length (cm)", sl)
st.write("sepal width (cm)", sw)
st.write("petal length (cm)", pl)
st.write("petal width (cm)", pw)

prediccion = pd.DataFrame ({
  "sepal length (cm)": sl,
  "sepal width (cm)": sw,
  "petal length (cm)": pl,
  "petal width (cm)": pw
}, index=[0])


y_predicted = model.predict (prediccion)

st.write("Prediccion de flor:", iris.target_names[y_predicted[0]])