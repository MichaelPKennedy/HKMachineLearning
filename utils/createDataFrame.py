from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd
import os

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


queries = {
    "ML_Demographics": "SELECT d.*, c.area_code FROM ML_Demographics d JOIN City c ON d.city_id = c.city_id WHERE d.population IS NOT NULL",
    "ML_Housing": "SELECT * FROM ML_Housing WHERE rent_price IS NOT NULL AND home_price IS NOT NULL",
    "ML_Industry": "SELECT * FROM ML_Industry",
    "ML_RecreationAndScenery": "SELECT * FROM ML_RecreationAndScenery",
    "ML_Weather": "SELECT * FROM ML_Weather",
}

dataframes = {name: pd.read_sql_query(query, engine) for name, query in queries.items()}

combined_df = dataframes['ML_Demographics']
for name, df in dataframes.items():
    if name != 'ML_Demographics': 
        combined_df = combined_df.merge(df, on='city_id', how='inner')

# combined_df.to_pickle('AllDataFrames.pkl')

combined_df.to_csv('combined_data.csv', index=False)

