from fastapi import FastAPI
from fastapi.responses import JSONResponse 
app = FastAPI()


# Route For Server Health Check
@app.get("/")
async def router_health():
    return JSONResponse(content={
        "message": "server is up and running",
        "healthy": True
    }, status_code=200)