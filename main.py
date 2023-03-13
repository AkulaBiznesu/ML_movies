import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

path = "/Users/Bogruk/Downloads/archive/movies_metadata.csv"
metadata = pd.read_csv(path, low_memory=False)
# print(metadata["overview"].head())

tfidf = TfidfVectorizer(stop_words="english")
metadata["overview"] = metadata["overview"].fillna("")
tfidf_matrix = tfidf.fit_transform(metadata["overview"])
tfidf_matrix.shape

tfidf.get_feature_names_out()[5000:5010]

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim.shape

indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
# print(indices[:10])

def get_recomendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices= [i[0] for i in sim_scores]
    return metadata["title"].iloc[movie_indices]

get_recomendations("The Godfather")