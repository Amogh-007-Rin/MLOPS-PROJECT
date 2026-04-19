## TASK-01 ##
    - Navigate to the airflow/dags folder 
    - Find the csv_to_postgres_dag.py file
    - Create a airflow dag which migrates the data from the dataset[csv file] present in the Dataset/Raw folder to the postgres database running locally on this machine. for reference look into the docker-compose.yml file present in the root folder.
    - Result: The csv_to_postgres dag must successfully load the data present in the dataset over to the postgres database.

## TASK-01 FINISHED NOTES ##

### Changes Made

1. **`docker-compose.yml`**
   - Updated Airflow dags volume mount: `./dags` → `./airflow/dags`
   - Added `./Dataset:/opt/airflow/dataset` volume so the CSV is accessible inside Airflow containers

2. **`airflow/dags/csv_to_postgres_dag.py`**
   - DAG ID: `csv_to_postgres`, scheduled `@once`, catchup disabled
   - Task 1 — `create_table`: Creates the `NearEarthObject` table in Postgres if it doesn't exist
   - Task 2 — `load_csv_to_postgres`: Reads `Dataset/Raw/neo.csv` via pandas and upserts all rows into `NearEarthObject` using `ON CONFLICT (id) DO UPDATE`

### NearEarthObject Table Schema

| Column           | Type         |
|------------------|--------------|
| id               | BIGINT (PK)  |
| name             | TEXT         |
| est_diameter_min | FLOAT        |
| est_diameter_max | FLOAT        |
| relative_velocity| FLOAT        |
| miss_distance    | FLOAT        |
| orbiting_body    | TEXT         |
| sentry_object    | BOOLEAN      |
| absolute_magnitude | FLOAT      |
| hazardous        | BOOLEAN      |

### Required Airflow Connection (add via Admin → Connections)

| Field           | Value         |
|-----------------|---------------|
| Connection ID   | `neo_postgres` |
| Connection Type | `Postgres`    |
| Host            | `postgres`    |
| Schema          | `mlops`       |
| Login           | `mlops`       |
| Password        | `password123` |
| Port            | `5432`        |

### How to Run

1. Start services: `docker-compose up -d`
2. Add the `neo_postgres` connection in Airflow UI (`localhost:8080`)
3. Trigger the `csv_to_postgres` DAG manually
4. Both tasks (`create_table` → `load_csv_to_postgres`) must turn green for success
