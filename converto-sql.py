import sqlite3

sqlite_db = "wells.db"

# save results to sqlite
conn = sqlite3.connect(sqlite_db)
results.to_sql("wells", conn, if_exists="replace", index=False)
conn.close()
print(f"Saved {len(results)} wells to {sqlite_db}")