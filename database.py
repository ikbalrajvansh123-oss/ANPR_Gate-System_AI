import sqlite3

class DB:
    def __init__(self):
        self.conn = sqlite3.connect("data.db", check_same_thread=False)
        self.cur = self.conn.cursor()

        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS logs(
            number TEXT,
            datetime TEXT
        )
        """)
        self.conn.commit()

    def insert(self, plate, dt):
        self.cur.execute(
            "INSERT INTO logs VALUES (?, ?)",
            (plate, dt)
        )
        self.conn.commit()

    def fetch(self):
        self.cur.execute("SELECT * FROM logs ORDER BY datetime DESC")
        return self.cur.fetchall()

    def delete_all(self):
        self.cur.execute("DELETE FROM logs")
        self.conn.commit()