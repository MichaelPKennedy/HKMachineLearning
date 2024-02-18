
#example of how trained model would be used to make inferences on new data
def predict_interest(model, user_vector, all_city_features):
    """
    Predicts a user's interest in all cities.
    
    Parameters:
    - model: The trained FM model.
    - user_vector: The preference vector for the user.
    - all_city_features: A matrix of feature vectors for all cities.
    
    Returns:
    A list or array of scores indicating the user's predicted interest in each city.
    """
    # Ensure user_vector is repeated for each city for batch processing
    user_vectors = user_vector.repeat(all_city_features.shape[0], 1)
    
    # Model predictions
    interest_scores = model(user_vectors, all_city_features)
    
    return interest_scores
