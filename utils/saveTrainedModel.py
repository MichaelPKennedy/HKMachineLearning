# save_model.py
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from joblib import dump

# Assuming the DataFrame is loaded similar to the original script
combined_df = pd.read_pickle("AllDataFrames.pkl")
X = combined_df.drop('city_id', axis=1)

knn = NearestNeighbors(n_neighbors=50)
knn.fit(X)

# Save the trained model to a file
dump(knn, 'knn_model.joblib')

dump(combined_df, 'combined_df.joblib')
