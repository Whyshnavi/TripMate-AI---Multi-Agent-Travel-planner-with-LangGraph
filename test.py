
import os
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

print("Testing PostgreSQL connection pool...")

pool = None

try:
    pool = ConnectionPool(
        conninfo=DATABASE_URL,
        min_size=1,
        max_size=1,
        num_workers=1,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "connect_timeout": 15,
        },
        open=False,
    )

    print("Opening connection pool...")

    pool.open(
        wait=True,
        timeout=25,
    )

    print("Connection pool opened successfully!")

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            result = cur.fetchone()

            print("\nSUCCESS!")
            print("Database:", result["version"])

except Exception as e:
    print("\nPOOL CONNECTION FAILED")
    print("Error type:", type(e).__name__)
    print("Error:", str(e))

finally:
    if pool is not None:
        pool.close()
        print("Connection pool closed.")
