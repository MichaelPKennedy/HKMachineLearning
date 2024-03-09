import torch

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
