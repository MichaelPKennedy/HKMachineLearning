const mysql = require("mysql2/promise");
const dotenv = require("dotenv");

dotenv.config();

const dbConfig = {
  host: process.env.host,
  user: process.env.username,
  password: process.env.password,
  database: process.env.database,
  port: process.env.port,
};

exports.updateTopCities = async (req, res) => {
  const connection = await mysql.createConnection(dbConfig);
  try {
    const now = new Date();
    const currentMonth = now.getMonth() + 1;

    const firstDayOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
    const formattedFirstDayOfMonth = firstDayOfMonth
      .toISOString()
      .split("T")[0];

    const getTopMonthlyCitiesQuery = `
            SELECT c.city_id, COUNT(u.city_id) AS city_count
            FROM UserCities u
            JOIN City c ON u.city_id = c.city_id
            WHERE u.createdAt >= ?
            GROUP BY u.city_id
            ORDER BY city_count DESC
            LIMIT 20;
        `;

    const getTopCitiesAllTimeQuery = `
            SELECT c.city_id, COUNT(u.city_id) AS city_count
            FROM UserCities u
            JOIN City c ON u.city_id = c.city_id
            GROUP BY u.city_id
            ORDER BY city_count DESC
            LIMIT 20;
        `;

    const [topMonthlyCities] = await connection.execute(
      getTopMonthlyCitiesQuery,
      [formattedFirstDayOfMonth]
    );
    const [topCitiesAllTime] = await connection.execute(
      getTopCitiesAllTimeQuery
    );

    await Promise.all(
      topMonthlyCities.map((city, index) =>
        connection.execute(
          `INSERT INTO TopMonthlyCities (ranking, city_id, count, month, updatedAt) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                 ON DUPLICATE KEY UPDATE city_id = ?, month= ?, count = ?, updatedAt = CURRENT_TIMESTAMP;`,
          [
            index + 1,
            city.city_id,
            city.city_count,
            currentMonth,
            city.city_id,
            currentMonth,
            city.city_count,
          ]
        )
      )
    );

    await Promise.all(
      topCitiesAllTime.map((city, index) =>
        connection.execute(
          `INSERT INTO TopCities (ranking, city_id, count, updatedAt) VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                 ON DUPLICATE KEY UPDATE city_id = ?, count = ?, updatedAt = CURRENT_TIMESTAMP;`,
          [
            index + 1,
            city.city_id,
            city.city_count,
            city.city_id,
            city.city_count,
          ]
        )
      )
    );

    res.status(200).send("Top cities statistics updated successfully.");
  } catch (error) {
    console.error("Failed to update top cities statistics:", error);
    res.status(500).send("Error updating top cities statistics");
  } finally {
    await connection.end();
  }
};
