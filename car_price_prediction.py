# Car Price Prediction - FINAL FIXED VERSION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("car data.csv")

print("\nColumns in Dataset:")
print(data.columns)

# -------------------------------
# 2. Clean Column Names
# -------------------------------
data.columns = data.columns.str.strip()

# -------------------------------
# 3. Drop Unnecessary Column
# -------------------------------
data.drop("Car_Name", axis=1, inplace=True)

# -------------------------------
# 4. Encode Categorical Columns
# -------------------------------

# Fuel Type
data["Fuel_Type"] = data["Fuel_Type"].map({
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2
})

# Selling Type (IMPORTANT FIX)
data["Selling_type"] = data["Selling_type"].map({
    "Dealer": 0,
    "Individual": 1
})

# Transmission
data["Transmission"] = data["Transmission"].map({
    "Manual": 0,
    "Automatic": 1
})

# -------------------------------
# 5. Feature Engineering
# -------------------------------
data["Car_Age"] = 2024 - data["Year"]
data.drop("Year", axis=1, inplace=True)

# -------------------------------
# 6. Handle Missing Values
# -------------------------------
data.dropna(inplace=True)

# -------------------------------
# 7. Feature & Target
# -------------------------------
X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

# -------------------------------
# 8. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 9. Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 10. Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 11. Evaluation
# -------------------------------
print("\nModel Performance:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------------
# 12. Visualization
# -------------------------------
plt.figure(figsize=(7, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Car Price Prediction (Linear Regression)")
plt.show()
