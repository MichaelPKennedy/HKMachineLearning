import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import functions_framework
from google.cloud import storage
from joblib import load
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
import warnings
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
        print(f"An error occurred: {e}")
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

def send_email_to_user(email, user_id, name):
    sg = SendGridAPIClient(api_key=os.getenv('SENDGRID_API_KEY'))

    from_email = Email('michael@homeknown.app', 'HomeKnown')
    to_email = To(email)
    subject = 'New AI Recommendations Available!'
    html_content = Content(
        'text/html', 
        f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>New AI Recommendations Available!</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; color: #444; }}
            .container {{ background-color: #fff; border: 1px solid #ddd; padding: 20px; max-width: 600px; margin: 20px auto; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; }}
            p {{ line-height: 1.6; }}
            .button {{ display: inline-block; padding: 10px 20px; margin: 10px 2px; border-radius: 5px; color: #FFFFFF !important; background-color: #01697c !important; text-decoration: none; }}
            .button:hover {{ background-color: #FFFFFF !important; color: #01697c !important; border: 1px solid #01697c; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 0.9em; color: #555; }}
        </style>
        </head>
        <body>
        <div class="container">
            <h1>Hello {name},</h1>
            <p>We have new personalized AI recommendations for you that we think you will love!</p>
            <p>Click the button below to view your recommendations:</p>
            <a href="https://www.homeknown.app/recommendations" class="button">Recommendations</a>
            <p>If you have any questions or need assistance, feel free to reach out to us:</p>
            <a href="https://www.homeknown.app/support" class="button">Contact Support</a>
        </div>
        </body>
        </html>
        '''
    )
    message = Mail(from_email, to_email, subject, html_content)

    try:
        response = sg.send(message)
        print(f"Email sent to user {user_id} ({email}) with status code {response.status_code}")
    except Exception as e:
        print(f"Error sending email to user {user_id}: {e}")



def update_user_recommendations_with_transaction(engine, combined_df, knn):
    one_hour_ago = datetime.now() - timedelta(hours=1)
    twenty_minutes_ago = datetime.now() - timedelta(minutes=20)
    updated_users = []

    with engine.connect() as connection:
        trans = connection.begin()
        try:
            fetch_users_sql = text("""
                SELECT DISTINCT uc.user_id, u.primary_email, u.first_name, u.username 
                FROM UserCities uc
                JOIN Users u ON uc.user_id = u.user_id
                WHERE createdAt >= :twenty_minutes_ago
            """)

            # Pass parameters in a dictionary
            users = connection.execute(fetch_users_sql, {'twenty_minutes_ago': twenty_minutes_ago}).fetchall()
            print(f"Updating recommended locations for {len(users)} users")
            for user in users:
                user_id, email, first_name, username = user[0], user[1], user[2], user[3]


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
                name = first_name if first_name else username
                send_email_to_user(email, user_id, name)

            trans.commit()
            print("Updated Users", updated_users)
        except SQLAlchemyError as e:
            trans.rollback()
            raise e