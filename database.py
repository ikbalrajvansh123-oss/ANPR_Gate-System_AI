import sqlite3
import logging

logger = logging.getLogger(__name__)


class DB:
    def __init__(self, path: str = "data.db"):
        try:
            self.conn = sqlite3.connect(path, check_same_thread=False)
            self.cur = self.conn.cursor()
            
            # Create table if not exists
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    number   TEXT NOT NULL,
                    datetime TEXT NOT NULL
                )
            """)
            self.conn.commit()
            logger.info(f"✅ Database initialized: {path}")
            
        except Exception as e:
            logger.error(f"❌ Database init failed: {e}")
            raise

    def insert(self, plate: str, dt: str):
        """Insert new plate entry"""
        try:
            self.cur.execute(
                "INSERT INTO logs (number, datetime) VALUES (?, ?)",
                (plate, dt)
            )
            self.conn.commit()
            logger.debug(f"Inserted: {plate}")
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            raise

    def fetch(self) -> list[tuple]:
        """Get all entries, newest first"""
        try:
            self.cur.execute(
                "SELECT number, datetime FROM logs ORDER BY datetime DESC"
            )
            return self.cur.fetchall()
        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            return []

    def delete_all(self):
        """Delete all entries"""
        try:
            self.cur.execute("DELETE FROM logs")
            self.conn.commit()
            logger.info("Deleted all logs")
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise

    def close(self):
        """Close connection"""
        try:
            self.conn.close()
            logger.info("Database closed")
        except Exception:
            pass