# 🏥 Insurance Premium Predictor (AI-Powered)

### 🚀 Overview

This is a Full-Stack Machine Learning application designed to predict annual medical insurance premiums based on user health metrics. It mimics real-world actuarial logic used by insurance companies to assess risk and calculate costs.

### 🛠️ Tech Stack

- **Frontend:** HTML5, CSS3 (Flexbox for responsive layout)
- **Backend:** Python (Flask)
- **Machine Learning:** Scikit-Learn (Linear Regression), Pandas, NumPy
- **Model Persistence:** Pickle

### 📊 How It Works

1.  **Data Processing:** The model takes user inputs (Age, BMI, Smoking Status).
2.  **Risk Assessment:** It applies a Linear Regression algorithm trained on medical cost datasets.
3.  **Prediction:** The backend calculates a risk score and returns the estimated premium in USD.

### 💡 Key Features

- **Real-time Inference:** Instant predictions via REST API.
- **Responsive UI:** Designed with a clean, medical-grade interface.
- **Risk Logic:** accurately weights high-risk factors like smoking (adds ~$20k penalty) and high BMI.

### 📸 Project Screenshot

![App Screenshot](path-to-your-image.png)
_(Upload your screenshot to the repo and link it here!)_
