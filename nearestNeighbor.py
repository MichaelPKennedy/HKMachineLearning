import pandas as pd
from sklearn.neighbors import NearestNeighbors


combined_df = pd.read_pickle("AllDataFrames.pkl")

X = combined_df.drop('city_id', axis=1)  # Features matrix
knn = NearestNeighbors(n_neighbors=50)
knn.fit(X)


def find_similar_cities(saved_city_ids, combined_df, knn_model):
    # Calculate the mean of the features of the saved cities
    saved_cities = combined_df[combined_df['city_id'].isin(saved_city_ids)]
    query_point = saved_cities.drop('city_id', axis=1).mean().to_frame().T
    # Use the KNN model to find the indices of the nearest neighbors
    distances, indices = knn_model.kneighbors(query_point, n_neighbors=50 + len(saved_city_ids))
    # Retrieve the city_ids of the recommended cities
    all_recommended_city_ids = combined_df.iloc[indices[0]]['city_id'].values
    # Exclude the saved city_ids from the recommendations
    recommended_city_ids = [city_id for city_id in all_recommended_city_ids if city_id not in saved_city_ids][:50]
    return recommended_city_ids


# Test
saved_city_ids = [20, 21]
recommended_cities = find_similar_cities(saved_city_ids, combined_df, knn)
print(recommended_cities)
