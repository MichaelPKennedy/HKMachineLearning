import os
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import functions_framework
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

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
def update_top_cities(request):
    connection = engine.connect()
    try:
        now = datetime.now()
        current_month = now.month

        first_day_of_month = datetime(now.year, now.month, 1)
        formatted_first_day_of_month = first_day_of_month.strftime('%Y-%m-%d')

        get_top_monthly_cities_query = """
            SELECT c.city_id, COUNT(u.city_id) AS city_count
            FROM UserCities u
            JOIN City c ON u.city_id = c.city_id
            WHERE u.createdAt >= :first_day_of_month
            GROUP BY u.city_id
            ORDER BY city_count DESC
            LIMIT 20;
        """

        get_top_cities_all_time_query = """
            SELECT c.city_id, COUNT(u.city_id) AS city_count
            FROM UserCities u
            JOIN City c ON u.city_id = c.city_id
            GROUP BY u.city_id
            ORDER BY city_count DESC
            LIMIT 20;
        """

        top_monthly_cities = connection.execute(text(get_top_monthly_cities_query), {'first_day_of_month': formatted_first_day_of_month}).fetchall()
        top_cities_all_time = connection.execute(text(get_top_cities_all_time_query)).fetchall()

        for index, city in enumerate(top_monthly_cities):
            connection.execute(
                text("""
                    INSERT INTO TopMonthlyCities (ranking, city_id, count, month, updatedAt)
                    VALUES (:ranking, :city_id, :count, :month, CURRENT_TIMESTAMP)
                    ON DUPLICATE KEY UPDATE city_id = :city_id, month = :month, count = :count, updatedAt = CURRENT_TIMESTAMP;
                """),
                {'ranking': index + 1, 'city_id': city.city_id, 'count': city.city_count, 'month': current_month}
            )

        for index, city in enumerate(top_cities_all_time):
            connection.execute(
                text("""
                    INSERT INTO TopCities (ranking, city_id, count, updatedAt)
                    VALUES (:ranking, :city_id, :count, CURRENT_TIMESTAMP)
                    ON DUPLICATE KEY UPDATE city_id = :city_id, count = :count, updatedAt = CURRENT_TIMESTAMP;
                """),
                {'ranking': index + 1, 'city_id': city.city_id, 'count': city.city_count}
            )

        connection.commit()
        return 'Top cities statistics updated successfully.', 200

    except SQLAlchemyError as error:
        connection.rollback()
        print("Failed to update top cities statistics:", error)
        return 'Error updating top cities statistics', 500

    finally:
        connection.close()
