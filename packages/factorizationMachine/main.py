import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import functions_framework
import torch
import torch.nn as nn
from google.cloud import storage
from joblib import load

# Assuming you have a function to load your FM model and city features
def load_model_and_city_features_from_gcs():
    """Loads the FM model and city features from Google Cloud Storage"""
    user_feature_dim = 35
    city_feature_dim = 19
    k = 10

    # Assuming city_features is loaded correctly and is a DataFrame
    city_features = load('combined_df.joblib')

    model = FMModel(user_feature_dim, city_feature_dim, k)
    model.load_state_dict(torch.load("fm_model.pth"))
    model.eval()

    # Convert DataFrame to tensor
    city_features_tensor = torch.tensor(city_features.values, dtype=torch.float)

    return model, city_features_tensor


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
    fetch_user_features_sql = text("""
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
    
    user_feature_row = connection.execute(fetch_user_features_sql, {'user_id': user_id}).fetchone()
    if user_feature_row:
        # Define min and max values for each feature based on your specifications
        min_max_pairs = [
            (0, 8),  # costOfLivingWeight
            (0, 8),  # recreationWeight
            (0, 8),  # weatherWeight
            (0, 8),  # sceneryWeight
            (0, 8),  # industryWeight
            (0, 8),  # publicServicesWeight
            (0, 8),  # crimeWeight
            (0, 8),  # airQualityWeight
            (0, 500000),  # job1Salary
            (0, 500000),  # job2Salary
            (0, 1),  # entertainment
            (0, 1),  # foodAndDrinks
            (0, 1),  # historyAndCulture
            (0, 1),  # beaches
            (0, 1),  # nature
            (0, 1),  # winterSports
            (0, 1),  # adventureAndSports
            (0, 1),  # wellness
            (0, 1),  # northeast
            (0, 1),  # midwest
            (0, 1),  # south
            (0, 1),  # west
            (0, 80),  # yearly_avg_temp_norm
            (0, 100),  # temp_variance_norm
            (0, 120),  # max_temp
            (-30, 80),  # min_temp
            (0, 50),  # precipitation
            (0, 50),  # snow
            (0, 1000001),  # pop_min
            (0, 10000000),  # pop_max
            (0, 1000000),  # homeMin
            (0, 1000000),  # homeMax
            (0, 5000),  # rentMin
            (0, 5000),  # rentMax
            (0, 100)  # humidity
        ]
        
        # Normalize each feature
        normalized_features = [
            normalize_feature(value, min_val, max_val) 
            for value, (min_val, max_val) in zip(user_feature_row, min_max_pairs)
        ]
        
        user_features_tensor = torch.tensor([normalized_features], dtype=torch.float)
        return user_features_tensor
    else:
        return None

@functions_framework.http
def update_recommendations(request):
    """
    HTTP Cloud Function to update user recommendations.
    """
    model, city_features = load_model_and_city_features_from_gcs()
    
    try:
        update_user_recommendations_with_transaction(engine, model, city_features)
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

def update_user_recommendations_with_transaction(engine, model, city_features):
    one_hour_ago = datetime.now() - timedelta(hours=1)
    
    with engine.connect() as connection:
        trans = connection.begin()
        try:
            # Fetch users with new surveys in the last hour
            fetch_users_sql = text("""
                SELECT DISTINCT user_id FROM UserSurveys
                WHERE createdAt >= :one_hour_ago
            """)
            users = connection.execute(fetch_users_sql, {'one_hour_ago': one_hour_ago}).fetchall()
            
            for user in users:
                user_id = user[0]
                user_vector = fetch_and_normalize_user_features(user_id, connection)
                if user_vector is None:
                    continue  # No survey data for this user

                interest_scores = predict_interest(model, user_vector, city_features)
                
                # Fetch already saved cities to exclude from recommendations
                saved_cities = connection.execute(text("""
                    SELECT city_id FROM UserCities WHERE user_id = :user_id
                """), {'user_id': user_id}).fetchall()
                saved_city_ids = {city[0] for city in saved_cities}

                # Assume city IDs correspond to indices in interest_scores + 1
                recommended_city_ids = [(user_id, i+1) for i, score in enumerate(interest_scores) if i+1 not in saved_city_ids][:50]  
                print(f"Updating recommendations for user {user_id}: {recommended_city_ids}")

                # Update the UserRecommendedCities table with new recommendations
                
            trans.commit()
        except SQLAlchemyError as e:
            print(f"Transaction failed: {e}")
            trans.rollback()

# Further implementation details for fetching user vectors, predicting with the FM model,
# filtering saved cities, and updating recommendations need to be completed.
