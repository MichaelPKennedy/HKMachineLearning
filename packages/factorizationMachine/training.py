import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam
from dotenv import load_dotenv

load_dotenv()

df_final_dataset = pd.read_csv('final_training_data.csv')


class UserCityDataset(Dataset):
    def __init__(self, user_features, city_features, targets):
        self.user_features = torch.tensor(user_features, dtype=torch.float)
        self.city_features = torch.tensor(city_features, dtype=torch.float)
        # Add an extra dimension to targets to match the model's output shape
        self.targets = torch.tensor(targets, dtype=torch.float).unsqueeze(-1)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            "user_features": self.user_features[idx],
            "city_features": self.city_features[idx],
            "targets": self.targets[idx]  # targets have shape [batch_size, 1]
        }


user_feature_columns = [
    'costOfLivingWeight', 'recreationWeight', 'weatherWeight', 
    'sceneryWeight', 'industryWeight', 'publicServicesWeight', 
    'crimeWeight', 'airQualityWeight', 'job1Salary', 'job2Salary',
    'user_interest_entertainment', 'user_interest_foodAndDrinks', 
    'user_interest_historyAndCulture', 'user_interest_beaches', 
    'user_interest_nature', 'user_interest_winterSports',
    'user_interest_adventureAndSports', 'user_interest_wellness', 
    'northeast', 'midwest', 'south', 'west','yearly_avg_temp_norm', 
    'temp_variance_norm', 'max_temp', 'precipitation', 'snow', 
    'min_temp', 'pop_min', 'pop_max', 'homeMin', 'homeMax', 
    'rentMin', 'rentMax', 'humidity']

city_feature_columns  = [
    'population', 'home_price', 'rent_price', 'average_salary', 
    'average_hourly','entertainment', 'beaches', 'riversAndLakes', 
    'mountains', 'adventureAndSports','historyAndCulture', 'nature', 
    'forests', 'winterSports', 'wellness','astronomy', 
    'yearly_avg_temp_norm', 'temp_variance_norm', 'avg_humidity'
]


user_features = df_final_dataset[user_feature_columns].values
city_features = df_final_dataset[city_feature_columns].values
targets = np.ones(len(df_final_dataset))  # All positive samples

dataset = UserCityDataset(user_features, city_features, targets)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

# Initialize model with actual dimensions
model = FMModel(user_feature_dim=len(user_feature_columns), city_feature_dim=len(city_feature_columns), k=10)

# Training loop
loss_function = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        predictions = model(batch["user_features"], batch["city_features"])
        loss = loss_function(predictions, batch["targets"])
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")


#example of how to save model
#torch.save(model.state_dict(), "fm_model.pth")