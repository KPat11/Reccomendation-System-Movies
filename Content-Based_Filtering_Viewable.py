# Converted notebook to py script so core code is viewable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

# Load Data
movies = pd.read_csv('data/movies.csv')
movies.head()

# ## Vectorize Descriptions

tfidf = TfidfVectorizer(stop_words='english')

# handling missing data
movies['overview'] = movies['overview'].fillna('')
movies['overview']

# Matrix
tfidf_matrix = tfidf.fit_transform(movies['overview'])

pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# Similarity Matrix
similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# Putting everything together
def similiarity_movies(movie_title, num_movies):
    idx = movies.loc[movies['title']==movie_title].index[0]
    
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    movie_indices = [tpl[0] for tpl in scores[1:num_movies + 1]]
    title = list(movies['title'].iloc[movie_indices])

    return title

# Can handle similar movies but needs to make more accurate predictions

# Similarity_model(title, num_movies)
# Title has to be exact ^        ^ Input number of movies you would recs on

similiarity_movies('Kung Fu Panda 3', 3)


