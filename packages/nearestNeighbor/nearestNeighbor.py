import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import functions_framework

from joblib import load

# Load the trained model 
knn = load('knn_model.joblib')
combined_df = load('combined_df.joblib')

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

@functions_framework.http
def update_recommendations(request):
    """
    HTTP Cloud Function that is triggered by HTTP request google scheduler job
    """
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
    print(recommended_city_ids)
    return recommended_city_ids


def update_user_recommendations_with_transaction(engine, combined_df, knn):
    one_hour_ago = datetime.now() - timedelta(hours=1)

    with engine.connect() as connection:
        trans = connection.begin()
        try:
            fetch_users_sql = text("""
                SELECT DISTINCT user_id FROM UserCities
                WHERE createdAt >= :one_hour_ago
            """)

            # Pass parameters in a dictionary
            users = connection.execute(fetch_users_sql, {'one_hour_ago': one_hour_ago}).fetchall()
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

            trans.commit()
        except SQLAlchemyError as e:
            trans.rollback()
            raise e