from fastapi import FastAPI

# 1. Create the app instance
app = FastAPI()

# 2. Define a "route" (endpoint)
@app.get("/")
def read_root():
    return {"status": "success", "message": "MLOps Server is Live!"}

# 3. Define a route with parameters
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id, "category": "ML-Model"}