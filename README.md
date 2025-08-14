💻 Laptop Price Prediction App

This project is a Machine Learning web application built using XGBoost and Streamlit that predicts the price of a laptop based on its specifications.

✨ The app provides a simple UI where users can select laptop features such as brand, type, CPU, GPU, RAM, storage, screen quality, etc., and get an estimated price in Euros (€).

🚀 Features

🎨 Interactive UI built with Streamlit

🤖 XGBoost model trained with 86% accuracy

🔤 LabelEncoder + custom mappings used for categorical features

📊 Colorful Graphs included:

Predicted Price vs Company Average Price

RAM Distribution (Pie Chart)

CPU Speed vs Price Trend (Line Chart)

📱 Sidebar inputs for clean modern look

💰 Realistic predictions (Budget laptops ~€500, Premium MacBooks ~€2500)

🛠️ Tech Stack

Python 3.11

Streamlit (UI)

XGBoost (ML Model)

Scikit-learn (Preprocessing)

Matplotlib / Pandas (Visualization)

📂 Project Structure
laptop-price-predictor/
│
├── app.py               # Main Streamlit application
├── laptop_model.pkl     # Trained XGBoost model
├── encoders.pkl         # Saved LabelEncoders & mappings
├── requirements.txt     # Python dependencies
└── README.md            # Project description

⚡ How to Run Locally

Clone the repository

git clone https://github.com/your-username/laptop-price-predictor.git
cd laptop-price-predictor


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py


Open in browser → http://localhost:8501

🌍 Deployment

Deploy easily on Streamlit Cloud (recommended for free hosting).

Just connect your GitHub repo with Streamlit Cloud and run.

🎯 Demo Predictions

Dell Notebook (8GB RAM, 256GB SSD, GTX 1050) → ~€1029

Apple MacBook Pro (16GB RAM, 512GB SSD, Retina, i7, Radeon Pro) → ~€2458

📌 Future Improvements

📊 Add more graphs & analytics

🌎 Currency conversion (€, $, ₹)

📱 Mobile responsive UI
