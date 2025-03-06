# House Price Prediction using Machine Learning

## 📌 Overview
This project implements a **House Price Prediction** model using **Random Forest Regressor**. The dataset used for training contains various house features like **Square Footage, Number of Bedrooms, Year Built, Lot Size, and Garage Size**. The model is trained, validated, optimized, and then deployed for predictions.

---
## 📂 Dataset
- **Dataset Name**: `house_price_regression_dataset.csv`
- **Features Used**:
  - `Square_Footage`
  - `Num_Bedrooms`
  - `Year_Built`
  - `Lot_Size`
  - `Garage_Size`
- **Target Variable**: `House_Price`

---
## 🛠️ Installation & Setup
### 1️⃣ Prerequisites
Ensure you have **Python** installed. Required libraries:
```bash
pip install pandas numpy scikit-learn tensorflow tabulate
```

### 2️⃣ Clone Repository
```bash
git clone https://github.com/Phinix-Morgan/house-price-prediction.git
cd house-price-prediction
```

### 3️⃣ Run the Jupyter Notebook
```bash
jupyter notebook
```

---
## 📊 Data Preprocessing
- **Check for missing values** and handle them.
- Convert `House_Price` from string to float if necessary.
- Select features and split into training & testing sets (80-20 split).

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
```

---
## 🤖 Model Training
A **Random Forest Regressor** is trained with the following configuration:
```python
from sklearn.ensemble import RandomForestRegressor
melbourne_model = RandomForestRegressor(random_state=1)
melbourne_model.fit(x_train, y_train)
```

---
## 📈 Model Evaluation
The model is evaluated using **Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score**.
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, val_predictions)
mse = mean_squared_error(y_test, val_predictions)
r2 = r2_score(y_test, val_predictions)
print(f"MAE: {mae}, MSE: {mse}, R² Score: {r2}")
```

---
## 🎯 Optimizing the Model
**Max Leaf Nodes** tuning is applied to find the best tree size:
```python
def get_mae(max_leaf_nodes, x_train, x_test, y_train, y_test):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(x_train, y_train)
    preds_val = model.predict(x_test)
    return mean_absolute_error(y_test, preds_val)
```
After finding the best `max_leaf_nodes`, the final model is trained and evaluated.

---
## 🏠 Predicting House Prices
New house price predictions can be made as follows:
```python
new_house = np.array([[4615, 4, 2000, 1.72, 1]])
new_house_df = pd.DataFrame(new_house, columns=['Square_Footage', 'Num_Bedrooms', 'Year_Built', 'Lot_Size', 'Garage_Size'])
predicted_price = final_model.predict(new_house_df)
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
```

---
## 💾 Model Saving & Deployment
The trained model is saved using **Pickle**:
```python
import pickle
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(final_model, f)
```
To download the model in Google Colab:
```python
from google.colab import files
files.download("house_price_model.pkl")
```

---
## 📌 Future Improvements
- Implement other regression models (Linear Regression, Decision Trees, etc.).
- Hyperparameter tuning using Grid Search.
- Deploy the model as a **web API** for real-time price predictions.

---
## 🤝 Contributing
Feel free to **fork** this repository and submit **pull requests**. Any improvements, feature additions, or bug fixes are welcome!

