import pandas as pd
from sqlalchemy import create_engine

# ---------------- EXTRACT ----------------
def extract():
    print("Extracting data...")
    df = pd.read_csv('/home/snehdeep/nasa-mlops-pipeline/data/raw/neo.csv')
    return df

# ---------------- TRANSFORM ----------------
def transform(df):
    print("Transforming data...")

    # Fill missing values
    df = df.ffill()

    # Lowercase column names
    df.columns = [col.lower() for col in df.columns]

    return df

# ---------------- LOAD ----------------
def load(df):
    print("Loading data into PostgreSQL...")

    from sqlalchemy import create_engine

    engine = create_engine(
        "postgresql://student:student123@localhost:5432/nasa_db"
    )

    # ✅ PASS ENGINE directly (NOT connection)
    df.to_sql(
        "neo_data",
        con=engine,
        if_exists="replace",
        index=False
    )

    print("Data loaded successfully!")
# ---------------- PIPELINE ----------------
def run_etl():
    try:
        df = extract()
        df = transform(df)
        load(df)
        print("ETL pipeline completed successfully!")
        return "success"
    except Exception as e:
        print(f"ETL failed: {e}")
        raise e

# ---------------- MAIN ----------------
if __name__ == "__main__":
    run_etl()
