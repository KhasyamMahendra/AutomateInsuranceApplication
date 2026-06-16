import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# 1. Create Synthetic Data (Mimicking real medical data)
# Logic: Base cost $2000 + ($250 * age) + ($300 * BMI) + ($20,000 if smoker)
data = {
    'age': np.random.randint(18, 65, 100),
    'bmi': np.random.uniform(18.5, 40, 100),
    'smoker': np.random.randint(0, 2, 100) # 0 = No, 1 = Yes
}
df = pd.DataFrame(data)

# Calculate 'charges' based on our logic + some random noise
df['charges'] = 2000 + (250 * df['age']) + (300 * df['bmi']) + (20000 * df['smoker']) + np.random.normal(0, 1000, 100)

# 2. Train the Model
X = df[['age', 'bmi', 'smoker']] # Features
y = df['charges']                # Target (Cost)

model = LinearRegression()
model.fit(X, y)

print("✅ Model trained successfully!")
print(f"Sample Prediction (Age: 30, BMI: 25, Non-Smoker): ${model.predict([[30, 25, 0]])[0]:.2f}")

# 3. Save the Model to a file
with open('insurance_model.pkl', 'wb') as file:
    pickle.dump(model, file)
    print("✅ Model saved as 'insurance_model.pkl'")