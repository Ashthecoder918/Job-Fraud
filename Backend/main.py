import torch
import pickle
import pandas as pd
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from structure_validation.provider import FraudDetector,Validation

MODEL_VERSION=1.0

# device determination
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@asynccontextmanager
async def lifespan(app:FastAPI):
    print("!!! Inference artifacts are loaded here !!!")

    with open('trained_model/scaler.pkl','rb') as f:
        scaler=pickle.load(f)
    model=FraudDetector(12)
    model.load_state_dict(torch.load('trained_model/fraud_detection_model.pt',map_location=device))
    model.to(device)
    model.eval()

# globally storing objects
    app.state.model=model 
    app.state.scaler=scaler

    yield # maintaining the server run on traffic
    print("!!! Inference stopped and cleaning instances!!!")

    del app.state.model
    del app.state.scaler

app=FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return JSONResponse({"message":"Welcome to advanced fraud job posting detection system."})

@app.get("/health")
def health():
    return JSONResponse({"status":"ok","version":MODEL_VERSION})

@app.post('/predict')
def prediction(request:Request,validation:Validation):
    try:
        model=request.app.state.model
        scaler=request.app.state.scaler

        data = [list(validation.model_dump().values())]
        scaled_data=scaler.transform(data)

        with torch.no_grad():
            pred= model(torch.tensor(scaled_data,dtype=torch.float32,device=device))

        return JSONResponse({'prediction':1 if pred.item()>=0.58 else 0 , "score": pred.item()})
    
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
