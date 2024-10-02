import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import functions_framework
import torch
import torch.nn as nn
from google.cloud import storage
from joblib import load
# import debugpy
import warnings

# prevent future warnings from cluttering CF logs
warnings.filterwarnings("ignore", category=FutureWarning)

storage_client = storage.Client()

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def load_model_from_gcs():
    """Loads the model and dataframe from Google Cloud Storage"""
    bucket_name = os.getenv('GCS_BUCKET') 
    model_blob_name = 'fm_model.pth'
    df_blob_name = 'combined_df.joblib'

    # Temporary paths within the Cloud Function environment
    model_temp_path = '/tmp/fm_model.pth'
    df_temp_path = '/tmp/combined_df.joblib'

    download_blob(bucket_name, model_blob_name, model_temp_path)
    download_blob(bucket_name, df_blob_name, df_temp_path)

    # Instantiate model
    user_feature_dim = 35
    city_feature_dim = 19
    k = 10
    model = FMModel(user_feature_dim, city_feature_dim, k)

    # Load the model state dict
    model_state = torch.load(model_temp_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    model.eval()  # Set the model to evaluation mode

    # Load the dataframe
    combined_df = load(df_temp_path)

    return model, combined_df


def load_model_and_city_features_from_gcs():
    """Loads the FM model and city features from Google Cloud Storage"""
    model, combined_df = load_model_from_gcs()

    city_features_df = combined_df
    city_features = city_features_df.drop('city_id', axis=1)

    # Convert DataFrame to tensor
    city_features_tensor = torch.tensor(city_features.values, dtype=torch.float)

    return model, city_features_tensor, city_features_df


load_dotenv()

db_config = {
    "host": os.getenv("host"),
    "user": os.getenv("username"),
    "password": os.getenv("password"),
    "database": os.getenv("database"),
    "port": os.getenv("port"),
}

database_url = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
engine = create_engine(database_url)

# Factorization Machine layer
class FMLayer(nn.Module):
    def __init__(self, n, k):
        super(FMLayer, self).__init__()
        self.n = n  # Total number of features
        self.k = k  # Number of dimensions for factorization
        self.linear = nn.Linear(n, 1)
        self.V = nn.Parameter(torch.randn(n, k))

    def forward(self, x):
        linear_part = self.linear(x)
        interaction_part = 0.5 * (torch.pow(x @ self.V, 2) - (x**2) @ (self.V**2)).sum(1, keepdim=True)
        return linear_part + interaction_part

# Define the Factorization Machine model
class FMModel(nn.Module):
    def __init__(self, user_feature_dim, city_feature_dim, k):
        super(FMModel, self).__init__()
        self.fm_layer = FMLayer(user_feature_dim + city_feature_dim, k)
    
    def forward(self, user_features, city_features):
        combined_features = torch.cat((user_features, city_features), dim=1)
        return self.fm_layer(combined_features)
    

def normalize_feature(value, min_val, max_val):
    """Normalize a single feature value based on min and max values."""
    # Handle edge cases where min_val == max_val to avoid division by zero
    return (value - min_val) / (max_val - min_val) if max_val != min_val else 0

def fetch_and_normalize_user_features(user_id, connection):
    """Fetch user features for a given user_id, apply manual normalization, and return as a tensor."""
    query = text("""
    SELECT 
        costOfLivingWeight, 
        recreationWeight, 
        weatherWeight, 
        sceneryWeight, 
        industryWeight, 
        publicServicesWeight, 
        crimeWeight, 
        airQualityWeight, 
        job1Salary, 
        job2Salary, 
        entertainment as user_interest_entertainment, 
        foodAndDrinks as user_interest_foodAndDrinks, 
        historyAndCulture as user_interest_historyAndCulture, 
        beaches as user_interest_beaches, 
        nature as user_interest_nature, 
        winterSports as user_interest_winterSports, 
        adventureAndSports as user_interest_adventureAndSports, 
        wellness as user_interest_wellness, 
        yearly_avg_temp_norm as user_yearly_avg_temp_norm, 
        temp_variance_norm as user_temp_variance_norm, 
        max_temp, 
        min_temp, 
        precipitation, 
        snow, 
        pop_min, 
        pop_max, 
        northeast, 
        midwest, 
        south, 
        west, 
        homeMin, 
        homeMax, 
        rentMin, 
        rentMax, 
        humidity 
    FROM UserSurveys 
    WHERE user_id = :user_id 
    ORDER BY createdAt DESC 
    LIMIT 1
""")
    # TODO - Get the last 50 surveys, average the results, and then continue

    df_user_features = pd.read_sql_query(query, engine, params={'user_id': user_id})
    if not df_user_features.empty:
        # Predefined defaults for null values
        predefined_defaults = {
            'costOfLivingWeight': 0, 'recreationWeight': 0, 'weatherWeight': 0,
            'sceneryWeight': 0, 'industryWeight': 0, 'publicServicesWeight': 0,
            'crimeWeight': 0, 'airQualityWeight': 0, 'job1Salary': 50000,  # Default salary
            'job2Salary': 50000, 'user_interest_entertainment': 0, 'user_interest_foodAndDrinks': 0,
            'user_interest_historyAndCulture': 0, 'user_interest_beaches': 0, 'user_interest_nature': 0,
            'user_interest_winterSports': 0, 'user_interest_adventureAndSports': 0, 'user_interest_wellness': 0,
            'user_yearly_avg_temp_norm': 55, 'user_temp_variance_norm': 65, 'max_temp': 87,
            'min_temp': 22, 'precipitation': 50, 'snow': 50,
            'pop_min': 1, 'pop_max': 10000000, 'northeast': 0,
            'midwest': 0, 'south': 0, 'west': 0, 'homeMin': 150000,
            'homeMax': 600000, 'rentMin': 1200, 'rentMax': 1600, 'humidity': 66
        }

        # Apply defaults to null values
        for column, default in predefined_defaults.items():
            df_user_features[column].fillna(default, inplace=True)

        # Apply normalization directly based on predefined ranges
        normalization_ranges = {
            'costOfLivingWeight': (0, 8),
            'recreationWeight': (0, 8),
            'weatherWeight': (0, 8),
            'sceneryWeight': (0, 8),
            'industryWeight': (0, 8),
            'publicServicesWeight': (0, 8),
            'crimeWeight': (0, 8),
            'airQualityWeight': (0, 8),
            'job1Salary': (0, 500000),  
            'job2Salary': (0, 500000),  
            'user_interest_entertainment': (0, 1),  
            'user_interest_foodAndDrinks': (0, 1),  
            'user_interest_historyAndCulture': (0, 1), 
            'user_interest_beaches': (0, 1), 
            'user_interest_nature': (0, 1),  
            'user_interest_winterSports': (0, 1), 
            'user_interest_adventureAndSports': (0, 1), 
            'user_interest_wellness': (0, 1),
            'user_yearly_avg_temp_norm': (0, 80),
            'user_temp_variance_norm': (0, 100),
            'max_temp': (0, 120),
            'min_temp': (-30, 80),
            'precipitation': (0, 50),
            'snow': (0, 50),
            'pop_min': (0, 1000001),
            'pop_max': (0, 10000000),
            'northeast': (0, 1), 
            'midwest': (0, 1),
            'south': (0, 1),
            'west': (0, 1),
            'homeMin': (0, 1000000),
            'homeMax': (0, 1000000),
            'rentMin': (0, 5000),
            'rentMax': (0, 5000),
            'humidity': (0, 100)
        }

        for column, (min_val, max_val) in normalization_ranges.items():
            df_user_features[column] = df_user_features[column].apply(normalize_feature, args=(min_val, max_val))

        # Convert to tensor
        user_features_tensor = torch.tensor(df_user_features.values, dtype=torch.float).squeeze(0) 
        return user_features_tensor
    else:
        return None
   

@functions_framework.http
def update_recommendations(request):
    """
    HTTP Cloud Function to update user recommendations.
    """
 
    # needed to debug with functions-framework
    # debugpy.listen(5678)
    # debugpy.wait_for_client()
    # print("Debugger attached!")

    model, city_features, city_features_w_city_id = load_model_and_city_features_from_gcs()
    try:
        update_user_recommendations_with_transaction(engine, model, city_features, city_features_w_city_id)
        return 'Update process completed successfully.', 200
    except Exception as e:
        print(f"An error occurred: {e}")
        return 'Update process failed.', 500

def predict_interest(model, user_vector, all_city_features):
    """
    Predicts a user's interest in all cities.
    
    Parameters:
    - model: The trained FM model.
    - user_vector: The preference vector for the user, as a 1D PyTorch tensor.
    - all_city_features: A matrix of feature vectors for all cities, as a 2D PyTorch tensor.
    
    Returns:
    A numpy array of scores indicating the user's predicted interest in each city.
    """
    model.eval()  # Ensure the model is in evaluation mode
    
    # Check if inputs are tensors, convert if not
    if not isinstance(user_vector, torch.Tensor):
        user_vector = torch.tensor(user_vector, dtype=torch.float)
    if not isinstance(all_city_features, torch.Tensor):
        all_city_features = torch.tensor(all_city_features, dtype=torch.float)

    # Ensure user_vector is repeated for each city for batch processing
    user_vectors = user_vector.repeat(all_city_features.shape[0], 1)
    
    with torch.no_grad():  # Inference without tracking gradients
        interest_scores = model(user_vectors, all_city_features)
    
    # Convert to numpy for easier handling/consumption outside PyTorch
    interest_scores = interest_scores.squeeze().numpy()
    
    return interest_scores

def update_user_recommendations_with_transaction(engine, model, city_features_tensor, city_features_w_city_id):
    one_hour_ago = datetime.now() - timedelta(hours=1)
    updated_users = []

    with engine.connect() as connection:
        trans = connection.begin()
        try:
            fetch_users_sql = text("""
                SELECT DISTINCT user_id FROM UserSurveys
                WHERE createdAt >= :one_hour_ago
            """)
            users = connection.execute(fetch_users_sql, {'one_hour_ago': one_hour_ago}).fetchall()
            print(f"Updating recommended locations for {len(users)} users")
            
            for user in users:
                user_id = user[0]
                user_vector = fetch_and_normalize_user_features(user_id, connection)
                if user_vector is None:
                    continue

                interest_scores = predict_interest(model, user_vector, city_features_tensor)
                # Get indices of top recommended cities
                top_indices = np.argsort(interest_scores)[-500:] 

                # Map indices to city_ids
                recommended_city_ids = city_features_w_city_id.iloc[top_indices]['city_id'].values
                
                # Exclude saved cities
                saved_cities_query = text("""
                    SELECT city_id FROM UserCities WHERE user_id = :user_id
                """)
                saved_cities = connection.execute(saved_cities_query, {'user_id': user_id}).fetchall()
                saved_city_ids = {city[0] for city in saved_cities} 
                final_recommended_city_ids = [city_id for city_id in recommended_city_ids if city_id not in saved_city_ids][:50]

                delete_recommendations_sql = text("""
                    DELETE FROM UserRecommendedCities
                    WHERE user_id = :user_id
                    AND premium = 1
                """)

                connection.execute(delete_recommendations_sql, {'user_id':user_id})

                values_to_insert = [{'user_id': user_id, 'city_id': city_id} for city_id in final_recommended_city_ids]

                insert_statement = text("""
                    INSERT INTO UserRecommendedCities (user_id, city_id, premium)
                    VALUES (:user_id, :city_id, 1)
                """)

                connection.execute(insert_statement, values_to_insert)
                updated_users.append(user_id)

            trans.commit()
            print("Updated Users", updated_users)
        except SQLAlchemyError as e:
            trans.rollback()
            raise
