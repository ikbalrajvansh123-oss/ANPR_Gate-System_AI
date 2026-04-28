import sqlite3

class DB:
    def __init__(self, name):
        self.conn = sqlite3.connect(name, check_same_thread=False)
        self.cur = self.conn.cursor()

        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            number TEXT,
            entry TEXT,
            exit TEXT
        )
        """)
        self.conn.commit()

    # ---------------- ENTRY ----------------
    def insert_entry(self, number, entry_time):
        self.cur.execute(
            "INSERT INTO logs (number, entry, exit) VALUES (?, ?, ?)",
            (number, entry_time, None)
        )
        self.conn.commit()

    # ---------------- EXIT ----------------
    def update_exit(self, number, exit_time):
        self.cur.execute("""
        UPDATE logs
        SET exit = ?
        WHERE number = ? AND exit IS NULL
        """, (exit_time, number))
        self.conn.commit()

    # ---------------- FETCH ----------------
    def fetch(self):
        self.cur.execute("SELECT * FROM logs ORDER BY entry DESC")
        return self.cur.fetchall()

    # ---------------- DELETE ALL ----------------
    def delete_all(self):
        self.cur.execute("DELETE FROM logs")
        self.conn.commit()

    # ---------------- SEARCH ----------------
    def search_plate(self, number):
        self.cur.execute(
            "SELECT * FROM logs WHERE number LIKE ?",
            (f"%{number}%",)
        )
        return self.cur.fetchall()

    # ---------------- DELETE SPECIFIC ----------------
    def delete_plate(self, number):
        self.cur.execute(
            "DELETE FROM logs WHERE number=?",
            (number,)
        )
        self.conn.commit()