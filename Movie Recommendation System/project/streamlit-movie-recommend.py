import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data1=pd.read_csv(r"C:\Users\Admin\Downloads\movies.csv")

data1["YEAR"]=data1["YEAR"].str.replace("\D","")
data1["YEAR"]=data1["YEAR"].map(lambda x:list(str(x)[:4])).map(lambda x:"".join(x))
data1["STARS"]=data1["STARS"].str.replace("\n","")
data1["GENRE"]=data1["GENRE"].apply(lambda x:str(x).strip("\n"))
data1["ONE-LINE"]=data1["ONE-LINE"].apply(lambda x:str(x).strip("\n"))
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data1["ONE-LINE"])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices=pd.Series(data1.index, index=data1['MOVIES']).drop_duplicates()
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return data1['MOVIES'].iloc[movie_indices]
html_temp="""
          <div style="background-color:tomato;padding:10px">
          <h2 style="color:white;text-align:center;>Streamlit Movie recommendation Dashboard</h2>
          </div>
          """
st.title("Movie Recommendation System")
st.markdown(html_temp,unsafe_allow_html=True)
text=st.text_input("Enter movie Name","Sweet Tooth")
recommend=get_recommendations(text)
st.dataframe(recommend)