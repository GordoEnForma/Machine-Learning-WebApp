import pandas as pd
import streamlit as st
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# Generar Titulo

st.title("""
  Detector de Gordo de Gordos
  Detectar si alguien tiene diabetes usando ML y Python
""")

# Seleccionar el dataset y asignar a X y Y

X_data, y_data = load_diabetes(return_X_y=True,as_frame=True)

data_diabetes = load_diabetes()

X = data_diabetes.data
y = data_diabetes.target


df = pd.DataFrame(X_data,columns = X_data.keys())
df['target'] = y_data


if st.checkbox('Mostrar X & Y '):
    st.text('''
    El data set utilizado ha sido diabetes.csv de la libreria de ScikitLearn

    ''')
    st.header("Data: ")
    st.dataframe(X_data)
    st.header("Target Data: ")
    st.dataframe(y_data)

if st.checkbox('Mostar DataFrame'):
    st.header('DataSet: ')
    st.write(df)


# Asignar data a nuestras variables de entrenamiento y validacion

tamaño_test_data = st.sidebar.slider("Test Size: ",0.01,0.99,0.25)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=tamaño_test_data,random_state = 0)

class Train_Model():
    def __init__(self,x_train,y_train,modelo):
        self.x_train= x_train
        self.y_train = y_train
        self.modelo = modelo

    def entrenar(self):
        self.modelo.fit(self.x_train,self.y_train)


classifier_name = st.sidebar.selectbox("Seleccione el Clasificador: ",
                                        ("RandomForestRegressor",
                                        "LinearRegression",
                                        "KNeighborsRegressor",
                                        "LogisticRegression",
                                        "DecisionTreeRegressor",
                                        ))

def get_Model(dataset_name):
    model = object
    if(dataset_name == "RandomForestRegressor"):
        model = RandomForestRegressor()
    elif(dataset_name == "LinearRegression"):
        model = LinearRegression()
    elif(dataset_name == "KNeighborsRegressor"):
        model = KNeighborsRegressor()
    elif(dataset_name == "DecisionTreeRegressor"):
        model = DecisionTreeRegressor()
    elif(dataset_name == "LogisticRegression"):
        model = LogisticRegression()
    return model

def get_score():
    selected_model = get_Model(classifier_name)
    modelazo = Train_Model(X_train,y_train,selected_model )
    modelazo.entrenar()
    score = modelazo.modelo.score(X_test,y_test)
    return score

st.header(classifier_name + ' Accuracy Score: ')
st.write(get_score())


# ModeloKN = Train_Model(X_train,y_train,KNeighborsRegressor())
# ModeloKN.entrenar()
# st.write(ModeloKN.modelo.score(X_test,y_test))

# ModeloRF = Train_Model(X_train,y_train,RandomForestRegressor(n_estimators=100) )
# ModeloRF.entrenar()
# st.write(ModeloRF.modelo.score(X_test,y_test))

# ModeloRT = Train_Model(X_train,y_train,DecisionTreeRegressor() )
# ModeloRT.entrenar()
# st.write(ModeloRT.modelo.score(X_test,y_test))

# ModeloLR = Train_Model(X_train,y_train,LinearRegression() )
# ModeloLR.entrenar()
# st.write(ModeloLR.modelo.score(X_test,y_test))

# ModeloLoR = Train_Model(X_train,y_train,LogisticRegression() )
# ModeloLoR.entrenar()
# st.write(ModeloLoR.modelo.score(X_test,y_test))


