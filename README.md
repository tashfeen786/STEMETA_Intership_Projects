# STEMETA_Intership_Projects

# 🚢 Titanic Survival Prediction API

A simple and efficient **FastAPI** application that predicts whether a passenger survived the Titanic disaster based on various features. This project demonstrates how to build a Machine Learning model and deploy it using an API.

---

## 📌 Features

- 🔍 Predict survival based on passenger data (age, sex, class, fare, etc.)
- ⚡ Built with **FastAPI** for high-performance API development
- 📦 Model trained using **scikit-learn** and saved via **pickle**
- 🔁 Easily extendable and ready for frontend integration
- 🔧 API reloads automatically using Uvicorn's `--reload` mode

---

## 📂 Project Structure

TitanicAPI/
├── Titanic.py # FastAPI main app
├── model.pkl # Trained machine learning model
├── requirements.txt # Python dependencies
└── README.md # Project documentation
✅ requirements.txt
Install dependencies
pip install -r requirements.txt

✅ Model Details
Algorithm: Logistic Regression / Random Forest (replace with actual model)
Trained on: Kaggle's Titanic Datase
Preprocessing: Label Encoding, Imputation, Scaling

💡 Future Improvements
Add Docker support for containerized deployment
Connect with a Streamlit or React frontend
Add authentication for secure API use
Deploy on cloud (e.g., Heroku, Render, or AWS)

👨‍💻 Author
Tashfeen Aziz
Data Analyst | Python Developer | ML/DL Enthusiast

GitHub: https://github.com/tashfeen786
LinkedIn: https://www.linkedin.com/in/tashfeen-aziz-b51361292/  
