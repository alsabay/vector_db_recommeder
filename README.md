The provided Jupyter notebook demonstrates building a movie recommendation system using vector databases, Word2Vec embeddings, and Faiss for efficient similarity search. 

The notebook showcases how to leverage Word2Vec embeddings, Faiss indexing, and user profiles to create a movie recommendation system with personalized recommendations. It demonstrates the power of vector databases and embeddings in making similarity-based recommendations efficiently.

Here's a high-level description and explanation of the code:

1. **Loading Word2Vec Model and Dataset**: The notebook starts by importing necessary libraries, capturing messages (output), and loading a pre-trained Word2Vec model using the Gensim library. It also reads a movie dataset (Movies_dataset.csv) using pandas.
```python
%%capture _messages

import pandas as pd
import gensim.downloader as api
from gensim.models import Word2Vec
import re

# Load the pre-trained Word2Vec model or train your own on the dataset.
print("Loading Word2Vec model...")
model = api.load('word2vec-google-news-300')
# Load the Movies_dataset.csv using pandas.
df = pd.read_csv('./data/Movies_dataset.csv')
```
2. **Preprocessing Text Data**: The 'Movie_Name' column in the dataset contains movie titles that need to be preprocessed before using them as input to the Word2Vec model. The function `preprocess_text` is defined to remove non-alphanumeric characters and convert the text to lowercase. The cleaned movie titles are then stored in a new column called 'title_cleaned'.
```python
def preprocess_text(text):
    # Remove non-alphanumeric characters and convert to lowercase.
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return text

# Apply the preprocessing to the "Movie_Name" column.
df['title_cleaned'] = df['Movie_Name'].apply(preprocess_text)
```

3. **Removing Duplicate Movie Titles**: Some movie titles might have duplicates in the dataset. To ensure each movie is represented only once, duplicates are removed based on the 'title_cleaned' column.
```python
# Remove duplicates (if any) based on the "title_cleaned" column.
df = df.drop_duplicates(subset='title_cleaned', keep='first')
```
4. **Creating Item Vectors**: The function `item_name_to_vector` converts movie titles to their vector representations using the Word2Vec model. For each movie in the dataset, a dictionary 'item_vectors' is created, where the key is the movie title, and the value is its corresponding vector. Movie titles not found in the Word2Vec vocabulary are excluded from the 'item_vectors' dictionary.
```python
# Function to convert item names to vectors using the Word2Vec model.
def item_name_to_vector(item_name):
    try:
        return model[item_name]
    except KeyError:
        return None

# Create a dictionary to store item vectors.
item_vectors = {item: item_name_to_vector(item) for item in movies}

# Remove items with no corresponding vectors (not in the Word2Vec vocabulary).
item_vectors = {item: vector for item, vector in item_vectors.items() if vector is not None}
```

5. **Setting up Faiss Index**: The 'item_vectors' are then converted to a NumPy array, which is used to initialize a Faiss index with L2 (Euclidean) distance metric. The item vectors are added to the Faiss index, and the index is saved to a file for future use.
```python
# Convert item vectors to a NumPy array for Faiss indexing.
item_vector_array = [vector for vector in item_vectors.values()]
item_vector_array = [vector.tolist() for vector in item_vector_array]

# Convert the list of lists into a 2D NumPy array.
item_vector_array = np.array(item_vector_array, dtype='float32')

# Initialize a Faiss index.
index = faiss.IndexFlatL2(model.vector_size)

# Add the item vectors to the index.
index.add(item_vector_array)

# Save the index to a file for future use.
faiss.write_index(index, './data/movie_index.faiss')
```
6. **Getting Similar Movie Recommendations**: A function called `find_similar_items` is defined to get similar movie recommendations based on user input. Users can input their preferences or interests, and the function uses Faiss to find similar movies to the user's input. The similarity is based on the Word2Vec vectors of the movie titles. The function then displays the top-k similar movie recommendations.
```python
# Function to get user input and find similar items.
def find_similar_items(user_input, k=5):
    # Preprocess the user input to match the format of item names in "item_vectors".
    preprocessed_input = preprocess_text(user_input)
    
    # Get the vector representation of the user input.
    input_vector = item_name_to_vector(preprocessed_input)
    
    if input_vector is not None:
        # Convert the input vector to a 2D NumPy array.
        input_vector = np.array([input_vector], dtype='float32')
        
        # Perform similarity search using the Faiss index.
        _, indices = index.search(input_vector, k+1)  # +1 to exclude the input item itself from recommendations.
        
        # Get the names of the similar items.
        similar_items = [list(item_vectors.keys())[i] for i in indices[0]]
        
        # Exclude the input item from the recommendations.
        similar_items = [item for item in similar_items if item != preprocessed_input
```
7. **Adding User Profiles**: The notebook demonstrates how to add user profiles to make recommendations more personalized. A dictionary called 'user_profiles' is initialized to store liked items for each user. Functions `add_item_to_profile` and `get_user_recommendations` allow users to add items to their profiles and get personalized movie recommendations based on their profiles.

8. **Sample Usage**: The notebook provides sample usage examples to demonstrate how to use the recommendation system. It shows how to get recommendations for a general movie preference (using `find_similar_items`) and how to get personalized recommendations for a user (using `get_user_recommendations`) based on their liked movies.

```python
liked_items = ['epic', 'overcomer', ...]  # Add some item names that the user liked.
```
