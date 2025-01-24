import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from api.controllers import health_controller

app = FastAPI(version='2.0', title='Unsupervised Image Anomaly Detection API',
              description='API for training and evaluating anomaly detection and classification models on image '
                          'datasets')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    health_controller.router,
    prefix="/health",
    tags=["Health Check"],
    responses={404: {"description": "Not found"}},
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)


