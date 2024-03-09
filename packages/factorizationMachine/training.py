import pandas as pd
from sklearn.preprocessing import StandardScaler
# normalizing the user preferences
# `df_user` will be DataFrame containing user preferences
scaler = StandardScaler()
columns_to_normalize = ['costOfLivingWeight', 'recreationWeight', 'weatherWeight', 
                        'sceneryWeight', 'industryWeight', 'publicServicesWeight', 
                        'crimeWeight', 'airQualityWeight', 'yearly_avg_temp_norm', 
                        'temp_variance_norm', 'max_temp', 'min_temp', 'precipitation', 
                        'snow', 'pop_min', 'pop_max']
df_user[columns_to_normalize] = scaler.fit_transform(df_user[columns_to_normalize])


# Convert data into a format suitable for training
from torch.utils.data import Dataset, DataLoader
import torch

class UserCityDataset(Dataset):
    def __init__(self, user_features, city_features, targets):
        self.user_features = user_features
        self.city_features = city_features
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            "user_features": torch.tensor(self.user_features[idx], dtype=torch.float),
            "city_features": torch.tensor(self.city_features[idx], dtype=torch.float),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float)
        }

# Example usage
# Assuming `user_features`, `city_features`, and `targets` are your prepared numpy arrays
dataset = UserCityDataset(user_features, city_features, targets)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#FM layer
import torch
import torch.nn as nn
import torch.nn.functional as F

class FMLayer(nn.Module):
    def __init__(self, n, k):
        """
        n: Number of features.
        k: Number of dimensions for factorization.
        """
        super(FMLayer, self).__init__()
        self.n = n  # Number of features
        self.k = k  # Number of latent factors
        
        # Linear part
        self.linear = nn.Linear(n, 1)
        
        # Factorization part for pairwise interactions
        self.V = nn.Parameter(torch.randn(n, k))

    def forward(self, x):
        # Linear Terms
        linear_part = self.linear(x)
        
        # Pairwise Interactions
        # Efficient implementation of the pairwise interactions
        interaction_sum_square = torch.pow(torch.mm(x, self.V), 2)
        square_interaction_sum = torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))
        interaction_part = 0.5 * torch.sum(interaction_sum_square - square_interaction_sum, dim=1, keepdim=True)
        
        # Combining both parts
        output = linear_part + interaction_part
        return output



# Define the Factorization Machine model
class FMModel(nn.Module):
    def __init__(self, user_feature_dim, city_feature_dim, k):
        """
        user_feature_dim: Dimension of the user feature vector.
        city_feature_dim: Dimension of the city feature vector.
        k: Number of dimensions for factorization.
        """
        super(FMModel, self).__init__()
        self.fm_layer = FMLayer(user_feature_dim + city_feature_dim, k)
    
    def forward(self, user_features, city_features):
        # Combine user and city features
        combined_features = torch.cat((user_features, city_features), dim=1)
        
        # Pass combined features through the FM layer
        output = self.fm_layer(combined_features)
        return output


# Initialize model
model = FMModel(user_feature_dim=30, city_feature_dim=19, k=10)  # Adjust dimensions as necessary



# Define the Factorization Machine layer, training loop, and optimizer
from torch.optim import Adam

loss_function = nn.MSELoss()  # Or another appropriate loss function
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        user_features = batch["user_features"]
        city_features = batch["city_features"]
        targets = batch["targets"]
        
        # Forward pass
        predictions = model(user_features, city_features)
        loss = loss_function(predictions, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item()}")


#example of how to save model
#torch.save(model.state_dict(), "fm_model.pth")