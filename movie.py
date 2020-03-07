import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("all_movies.csv")
features = ['keywords', 'cast', 'genres', 'director']


def features_combine(row):
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']


for feature in features:
    df[feature] = df[feature].fillna('')  # filling all NaNs with blank string

df["combined_features"] = df.apply(features_combine,
                                   axis=1)  # applying combined_features() method over each rows of dataframe and storing the combined string in "combined_features" column

count_vec = CountVectorizer()  # create new CountVectorizer() object
count_matrix = count_vec.fit_transform(
    df["combined_features"])  # feed combined strings(movie contents) to CountVectorizer() object

cosine_similarity = cosine_similarity(count_matrix)


def get_title(index):
    return df[df.index == index]["title"].values[0]


def get_index(title):
    return df[df.title == title]["index"].values[0]


users_favourite = "Aliens"
movie_index = get_index(users_favourite)
similar_movies = list(enumerate(cosine_similarity[
                                    movie_index]))
sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:6]

i = 0
print("The movies that I recommend based on " + users_favourite + " are:\n")
for element in sorted_similar_movies:
    print(get_title(element[0]))
    i = i + 1
    if i > 5:
        break
