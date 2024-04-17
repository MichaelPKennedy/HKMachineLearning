import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import functions_framework
from google.cloud import storage
from joblib import load
# import debugpy
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
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
    model_blob_name = 'knn_model.joblib'
    df_blob_name = 'combined_df.joblib'

    # Temporary paths within the Cloud Function environment
    model_temp_path = '/tmp/knn_model.joblib'
    df_temp_path = '/tmp/combined_df.joblib'

    download_blob(bucket_name, model_blob_name, model_temp_path)
    download_blob(bucket_name, df_blob_name, df_temp_path)

    knn_model = load(model_temp_path)
    combined_df = load(df_temp_path)

    return knn_model, combined_df

load_dotenv()

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

@functions_framework.http
def update_recommendations(request):
    """
    HTTP Cloud Function that is triggered by HTTP request google scheduler job
    """
    # Load model and dataframe from Google Cloud Storage
    knn, combined_df = load_model_from_gcs()

    try:
        update_user_recommendations_with_transaction(engine, combined_df, knn)
        return 'Update process completed successfully.', 200
    except Exception as e:
        # print(f"An error occurred: {e}")
        return 'Update process failed.', 500


def find_similar_cities(saved_city_ids, combined_df, knn_model):
    # Calculate the mean of the features of the saved cities
    saved_cities = combined_df[combined_df['city_id'].isin(saved_city_ids)]
    query_point = saved_cities.drop('city_id', axis=1).mean().to_frame().T
    # Use the KNN model to find the indices of the nearest neighbors
    distances, indices = knn_model.kneighbors(query_point, n_neighbors=50 + len(saved_city_ids))
    # Retrieve the city_ids of the recommended cities
    all_recommended_city_ids = combined_df.iloc[indices[0]]['city_id'].values
    # Exclude the saved city_ids from the recommendations
    recommended_city_ids = [city_id for city_id in all_recommended_city_ids if city_id not in saved_city_ids][:50]

    return recommended_city_ids


def update_user_recommendations_with_transaction(engine, combined_df, knn):
    one_hour_ago = datetime.now() - timedelta(hours=1)
    twenty_minutes_ago = datetime.now() - timedelta(minutes=20)
    updated_users = []

    with engine.connect() as connection:
        trans = connection.begin()
        try:
            fetch_users_sql = text("""
                SELECT DISTINCT user_id FROM UserCities
                WHERE createdAt >= :twenty_minutes_ago
            """)

            # Pass parameters in a dictionary
            users = connection.execute(fetch_users_sql, {'twenty_minutes_ago': twenty_minutes_ago}).fetchall()
            print(f"Updating recommended locations for {len(users)} users")
            for user in users:
                user_id = user[0]


                fetch_saved_cities_sql = text("""
                    SELECT city_id FROM UserCities
                    WHERE user_id = :user_id
                """)

                saved_cities = connection.execute(fetch_saved_cities_sql, {'user_id':user_id}).fetchall()
                saved_city_ids = [city[0] for city in saved_cities]

                if not saved_city_ids:
                    continue

                recommended_city_ids = find_similar_cities(saved_city_ids, combined_df, knn)

                delete_recommendations_sql = text("""
                    DELETE FROM UserRecommendedCities
                    WHERE user_id = :user_id
                """)

                connection.execute(delete_recommendations_sql, {'user_id':user_id})

                values_to_insert = [{'user_id': user_id, 'city_id': city_id} for city_id in recommended_city_ids]

                insert_statement = text("""
                    INSERT INTO UserRecommendedCities (user_id, city_id)
                    VALUES (:user_id, :city_id)
                """)

                connection.execute(insert_statement, values_to_insert)
                updated_users.append(user_id)

            trans.commit()
            print("Updated Users", updated_users)
        except SQLAlchemyError as e:
            trans.rollback()
            raise e