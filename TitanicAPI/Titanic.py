# FastAPI aur zaruri libraries import karein
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder # LabelEncoder ko import karein
import numpy as np # numpy import kiya gaya hai agar model ko iski zaroorat ho

# FastAPI app ko initialize karein
app = FastAPI(
    title="Titanic Survived Prediction API",
    description="This API predicts survival on the Titanic based on passenger data.",
    version="1.0.0",
)

# CORS (Cross-Origin Resource Sharing) configuration
# Yeh zaruri hai taake aapka HTML frontend (jo shayad alag port ya domain par chal raha ho)
# aapke FastAPI backend se communicate kar sake.
origins = [
    "http://127.0.0.1:5500",  # Agar aap VS Code Live Server use kar rahe hain
    "http://localhost:5500",  # Localhost ke liye
    "null"                    # Local file access ke liye (browser mein file://) - production mein avoid karein
    # Yahan woh URL add karein jahan aapki HTML file host hogi
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Sabhi HTTP methods (GET, POST, etc.) ko allow karein
    allow_headers=["*"],  # Sabhi headers ko allow karein
)

# Model aur Scaler ko load karein
# Ensure these paths are correct relative to where your FastAPI app is run
try:
    # Model filename ko consistent rakhte hue 'linear_regression_model.pkl' use kiya gaya
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    print("Scaler loaded successfully!")
except FileNotFoundError:
    print("Error: Model or scaler file not found. Make sure 'linear_regression_model.pkl' and 'scaler.pkl' are in the same directory as your FastAPI app.")
    # Agar files nahi milti toh app ko exit kar dein ya error handle karein
    exit()

# Label Encoders ko load ya re-create karein
# HTML UI se aane wali string values ko encode karne ke liye
le_sex = LabelEncoder()
le_sex.fit(['male', 'female']) # Sex ki possible values

le_embarked = LabelEncoder()
le_embarked.fit(['S', 'C', 'Q']) # Embarked ki possible values

# Request body ke liye Pydantic model define karein
# Ab Sex aur Embarked string type ke honge jaisa HTML UI se aayega
class PassengerDetails(BaseModel):
    pclass: int
    sex: str        # 'male' or 'female'
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str   # 'S', 'C', or 'Q'

# Root route (optional, for testing if the server is running)
@app.get("/")
def read_root():
    return {"message": "Titanic Survival Prediction API is Live!"}

# Prediction route
@app.post("/predict")
async def predict_survival(passenger: PassengerDetails):
    """
    Passenger details leta hai aur Titanic survival ki prediction karta hai.
    """
    try:
        # Categorical features ko encode karein
        # HTML UI se aane wali string values ko integers mein convert karein
        sex_encoded = le_sex.transform([passenger.sex])[0]
        embarked_encoded = le_embarked.transform([passenger.embarked])[0]

        # Input features ko DataFrame mein convert karein
        # Column names aur order ko apne model ki training features ke mutabiq rakhein
        # Yeh order bahut zaruri hai!
        # Agar aapke model ne one-hot encoding use ki hai, toh yahan uske mutabiq columns banenge.
        # Simple Label Encoding ke liye, yeh columns kaafi hain.
        user_input_df = pd.DataFrame([[
            passenger.pclass,
            sex_encoded,
            passenger.age,
            passenger.sibsp,
            passenger.parch,
            passenger.fare,
            embarked_encoded
        ]], columns=['pclass', 'sex_encoded', 'age', 'sibsp', 'parch', 'fare', 'embarked_encoded'])
        # IMPORTANT: 'sex_encoded' and 'embarked_encoded' yahan sirf illustrative names hain.
        # Aapko apne model ke actual feature names use karne honge jo training ke waqt the.
        # Agar aapka model 'Sex_male' ya 'Embarked_S' jaise columns expect karta hai,
        # toh aapko yahan un columns ko banakar 0 ya 1 value deni hogi.

        # Numerical features ko scale karein
        # Scaler ko use karein jo training ke waqt fit kiya gaya tha
        scaled_input = scaler.transform(user_input_df)

        # Model se prediction karein
        prediction = model.predict(scaled_input)[0] # Prediction array se single value nikalen

        # Prediction ko integer mein convert karein (0 ya 1)
        return {"prediction": int(prediction)}

    except Exception as e:
        # Errors ko handle karein aur meaningful message return karein
        print(f"Prediction error: {e}") # Debugging ke liye server console par print karein
        return {"error": str(e), "message": "Prediction failed due to an internal server error. Please check the input data and server logs."}
