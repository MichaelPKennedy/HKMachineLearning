import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from dotenv import load_dotenv

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

# Query to fetch data
query = "SELECT costOfLivingWeight, recreationWeight, weatherWeight, sceneryWeight, industryWeight, publicServicesWeight, crimeWeight, airQualityWeight, job1, job2, job1Salary, job2Salary, entertainment, foodAndDrinks, historyAndCulture, beaches, nature, winterSports, adventureAndSports, wellness, yearly_avg_temp_norm, temp_variance_norm, max_temp, min_temp, precipitation, snow, pop_min, pop_max, northeast, midwest, south, west, homeMin, homeMax, rentMin, rentMax, humidity FROM UserSurveys;"

# Fetch data
df = pd.read_sql(query, engine)

# Specify columns for imputation and normalization
columns_to_impute_and_normalize = ['costOfLivingWeight', 'recreationWeight', 'weatherWeight', 
                                   'sceneryWeight', 'industryWeight', 'publicServicesWeight', 
                                   'crimeWeight', 'airQualityWeight', 'job1Salary', 'job2Salary',
                                   'yearly_avg_temp_norm', 'temp_variance_norm', 'max_temp', 
                                   'min_temp', 'pop_min', 'pop_max', 'homeMin', 'homeMax', 
                                   'rentMin', 'rentMax', 'humidity']

# Initialize the imputer
# Strategy can be "mean", "median", "most_frequent", or "constant"
imputer = SimpleImputer(strategy="median")

# Apply imputation
df[columns_to_impute_and_normalize] = imputer.fit_transform(df[columns_to_impute_and_normalize])

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Apply normalization
df[columns_to_impute_and_normalize] = scaler.fit_transform(df[columns_to_impute_and_normalize])

# Save to CSV
df.to_csv('normalized_user_surveys.csv', index=False)
