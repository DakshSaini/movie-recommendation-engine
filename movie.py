import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("all_movies.csv")
features = ['keywords', 'cast', 'genres', 'director']


def combine_features(row):
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']


for feature in features:
    df[feature] = df[feature].fillna('')

df["combined_features"] = df.apply(combine_features,
                                   axis=1)

count_vector = CountVectorizer()
count_matrix = count_vector.fit_transform(
    df["combined_features"])

cosine_similarity = cosine_similarity(count_matrix)


def get_title(index):
    return df[df.index == index]["title"].values[0]


def get_index(title):
    return df[df.title == title]["index"].values[0]


users_favourite = "Aliens"
movie_index = get_index(users_favourite)
similar_movies = list(enumerate(cosine_similarity[
    movie_index]))
sorted_similar_movies = sorted(
    similar_movies, key=lambda x: x[1], reverse=True)[1:6]

i = 0
print("Top 5 recommended movies" + users_favourite + " are:\n")
for element in sorted_similar_movies:
    print(get_title(element[0]))
    i = i + 1
    if i > 5:
        break
