import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
db_config = {
    "host": os.getenv("host"),
    "user": os.getenv("username"),
    "password": os.getenv("password"),
    "database": os.getenv("database"),
    "port": os.getenv("port"),
}

# Create database engine
database_url = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
engine = create_engine(database_url)

# Fetch user survey data
query_user_surveys = """
SELECT user_id, costOfLivingWeight, recreationWeight, weatherWeight, sceneryWeight, industryWeight, publicServicesWeight, crimeWeight, airQualityWeight, job1Salary, job2Salary, entertainment, foodAndDrinks, historyAndCulture, beaches, nature, winterSports, adventureAndSports, wellness, yearly_avg_temp_norm, temp_variance_norm, max_temp, min_temp, precipitation, snow, pop_min, pop_max, northeast, midwest, south, west, homeMin, homeMax, rentMin, rentMax, humidity 
FROM UserSurveys;
"""
df_user_surveys = pd.read_sql(query_user_surveys, engine)

# Specify columns for imputation and normalization
columns_to_process = [
    'costOfLivingWeight', 'recreationWeight', 'weatherWeight', 'sceneryWeight',
    'industryWeight', 'publicServicesWeight', 'crimeWeight', 'airQualityWeight',
    'job1Salary', 'job2Salary', 'entertainment', 'foodAndDrinks', 'historyAndCulture',
    'beaches', 'nature', 'winterSports', 'adventureAndSports', 'wellness', 'northeast',
    'midwest', 'south', 'west', 'yearly_avg_temp_norm', 'temp_variance_norm', 'max_temp',
    'min_temp', 'precipitation', 'snow', 'pop_min', 'pop_max', 'homeMin', 'homeMax',
    'rentMin', 'rentMax', 'humidity'
]

# Impute missing values
imputer = SimpleImputer(strategy="median")
df_user_surveys[columns_to_process] = imputer.fit_transform(df_user_surveys[columns_to_process])

# Normalize the data
scaler = MinMaxScaler()
df_user_surveys[columns_to_process] = scaler.fit_transform(df_user_surveys[columns_to_process])

# Fetch and merge liked cities
query_liked_cities = "SELECT user_id, city_id FROM UserCities;"
df_liked_cities = pd.read_sql(query_liked_cities, engine)
df_merged = pd.merge(df_user_surveys, df_liked_cities, on="user_id")

# Assuming the city features CSV is already normalized and ready to use
df_city_features = pd.read_csv('normalized_city_data.csv')

# Rename columns to differentiate user features from city features
df_merged.rename(columns={
    'historyAndCulture': 'user_interest_historyAndCulture',
    'beaches': 'user_interest_beaches',
    'nature': 'user_interest_nature',
    'winterSports': 'user_interest_winterSports',
    'adventureAndSports': 'user_interest_adventureAndSports',
    'wellness': 'user_interest_wellness',
    'entertainment': 'user_interest_entertainment',
    'foodAndDrinks': 'user_interest_foodAndDrinks',
    'yearly_avg_temp_norm': 'user_yearly_avg_temp_norm',
    'temp_variance_norm': 'user_temp_variance_norm',
    }, inplace=True)


# Merge city features
df_final_dataset = pd.merge(df_merged, df_city_features, on="city_id")

# Save the final DataFrame ready for model training
df_final_dataset.to_csv('final_training_data.csv', index=False)
