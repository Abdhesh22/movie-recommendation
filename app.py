from flask import Flask, render_template, redirect,request
import requests

app = Flask(__name__)

#content-based filtering 
import numpy as np
import pandas as pd

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits,on="title")

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.isnull().sum() 
movies.dropna(inplace=True)
movies.duplicated().sum()

import ast
def convert(obj):
    l = []
    #we have string of list so we have to convert it into the list for selecting the name from the list of dict so we have a function in module ast called literal_eval that convert string into the list
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert3(obj):
    l = []
    ct = 0
    for i in ast.literal_eval(obj):
        if ct == 3:
            break
        l.append(i['name'])
        ct+=1
    return l

movies['cast'] = movies['cast'].apply(convert3)

def fetch_director(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l

movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['cast']

new_df = movies[['movie_id','title','tags']]
#new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

import nltk 
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

from sklearn.feature_extraction.text import CountVectorizer
cv  = CountVectorizer(max_features = 5000,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
from sklearn.metrics.pairwise import cosine_similarity
#cosine_similarity(vectors).shape
#no of total distance = 4806 * 4806
similarity = cosine_similarity(vectors)

def fetch_detail(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=6c0595eb55d71ec798e8dc5bbe05b7e5&language='
                            'en-US'.format(movie_id))
    data = response.json()
    genreObj = data['genres']
    genre = genreObj[0]
    genre = genre['name']
    overview = data['overview']
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path'],overview,genre


def trailer(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}/videos?api_key=6c0595eb55d71ec798e8dc5bbe05b7e5&language='
                            'en-US'.format(movie_id))
    movieUrl = response.json()                          
    result = movieUrl['results']
    result = result[0]
    key = result['key']
    link = 'https://www.youtube.com/embed/{}?autoplay=1&autohide=2&border=0&wmode=opaque&enablejsapi=1&modestbranding=1&controls=0&showinfo=1&mute=1'.format(key)
    return link                         

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    # sorting in the reverse order
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
    
    recommend_movies = []
    recommend_movies_poster = []
    recommend_movies_overview = []
    recommend_links = []
    recommend_genre = []
    for i in movies_list:
        # fetching the movie name from the index
        recommend_movies.append(movies.iloc[i[0]].title)
        #fetch poster from TMDB's api
        movie_id = movies.iloc[i[0]].movie_id
        poster, overview,genre = fetch_detail(movie_id)
        recommend_links.append(trailer(movie_id))
        recommend_movies_poster.append(poster)
        recommend_movies_overview.append(overview)
        recommend_genre.append(genre)
    return recommend_movies,recommend_movies_poster,recommend_movies_overview,recommend_links,recommend_genre
#collabrative filtering
@app.route('/',methods = ['GET','POST'])
def index():
    movie='Avatar'
    if request.method == "POST":
        movie = request.form['name']
    try:
        names, posters, overview, links,genre = recommend(movie)
        return render_template('index.html',movie_name = names, poster_path = posters,movie = movie,overview = overview,links=links,genre = genre)
    except Exception as e:
        return redirect("/negative")

@app.route('/negative')
def neg():
    return render_template('negative.html')

if __name__ == '__main__':
    app.run(debug=True)