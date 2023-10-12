import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

st.write("""
        # simple Iris flower Prediction App
        # This App predict Iris flower types
        """)
st.sidebar.header("User Input parameters")

#user Input

def user_input_features():
    sepal_lengh = st.sidebar.slider('Sepal lengh', 4.3, 7.9, 5.4)# min, max, default
    sepla_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.0)# min, max, default
    petal_lengh = st.sidebar.slider('Petal lengh', 1.0, 6.9, 2.7)# min, max, default
    petal_width = st.sidebar.slider('Sepal lengh', 0.1, 2.5, 2.0)# min, max, default

    data = {
        'sepal_lengh ' : sepal_lengh,
        'sepla_width ' : sepla_width,
        'petal_lengh ' : petal_lengh,
        'petal_width ' : petal_width
    }

    features = pd.DataFrame(data, index=[0]) # Create dataframe with default values
    return features

df = user_input_features()
st.subheader("User Input Parameters")
st.write(df)



# Load Data

iris = datasets.load_iris()
X = iris.data
Y = iris.target
classes = iris.target_names
model = LinearDiscriminantAnalysis()
model.fit(X, Y) # training
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

#display
st.subheader('Classes')
st.write(pd.array(classes))
st.subheader('Prediction')
st.write(prediction)
st.write(classes[prediction])
st.subheader('Probability')
st.write(prediction_proba)
