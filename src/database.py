from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from config import settings

# For MySQL, set DATABASE_URL to e.g. mysql+pymysql://user:pass@localhost:3306/aivoa_complaints
engine = create_engine(settings.database_url, pool_pre_ping=True,connect_args={"sslmode": "require"})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# 5. Added: Function to execute your specific query using the setup above
def check_db_version():
    # Use the engine directly for simple script execution without needing a full ORM session
    with engine.connect() as conn:
        result = conn.execute(text('SELECT VERSION()'))
        version = result.fetchone()[0]
        print(f"Database Version: {version}")

# Example of how to run it
if __name__ == "__main__":
    check_db_version()
