# House Price Prediction using Machine Learning

A Machine Learning web application that predicts house prices based on property features using a **Random Forest Regressor**. The project covers the complete ML workflow, including data preprocessing, model training, evaluation, optimization, model serialization, and deployment through a Streamlit web application.

---

## Features

- Predict house prices using a trained Random Forest Regressor.
- Interactive Streamlit web interface.
- Data preprocessing and feature selection.
- Train/Test data splitting.
- Model evaluation using regression metrics.
- Hyperparameter optimization using `max_leaf_nodes`.
- Model serialization using Pickle.
- Easy-to-use prediction interface.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle
- Jupyter Notebook

---

## Dataset

**Dataset:** `house_price_regression_dataset.csv`

### Input Features

- Square Footage
- Number of Bedrooms
- Year Built
- Lot Size
- Garage Size

### Target Variable

- House Price

---

## Machine Learning Workflow

```

Dataset
↓
Data Exploration
↓
Data Preprocessing
↓
Feature Selection
↓
Train/Test Split
↓
Random Forest Training
↓
Model Evaluation
↓
Hyperparameter Optimization
↓
Save Model (.pkl)
↓
Streamlit Deployment

```

---

## Model

The project uses a **Random Forest Regressor** from Scikit-learn to estimate house prices.

The model is optimized by experimenting with different values of `max_leaf_nodes` before training the final model.

---

## Model Evaluation

The model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

These metrics help measure prediction accuracy and overall model performance.

---

## Project Structure

```

house-price-prediction/
│
├── DATASET/
│   └── house_price_regression_dataset.csv
│
├── JUPYTER FILE/
│   └── house_prediction_kaggle.ipynb
│
├── PYTHON FILE/
│   └── house_prediction_kaggle.py
│
├── MAIN APP/
│   ├── streamlit_app.py
│   └── house_price_model_kaggle.pkl
│
├── requirements.txt
└── README.md

```

---

## Installation

Clone the repository.

```bash
git clone https://github.com/Phinix-Morgan/house-price-prediction.git
cd house-price-prediction
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## Running the Application

Navigate to the application directory.

```bash
cd "MAIN APP"
```

Run the Streamlit application.

```bash
streamlit run streamlit_app.py
```

The application will open in your browser, where you can enter property details and receive an estimated house price.

---

## Training the Model

The training notebook includes:

- Data loading
- Data exploration
- Missing value analysis
- Feature selection
- Model training
- Performance evaluation
- Hyperparameter optimization
- Model saving

You can retrain the model using either:

- `house_prediction_kaggle.ipynb`
- `house_prediction_kaggle.py`

---

## Example Prediction

Input

| Feature | Value |
|---------|------:|
| Square Footage | 4615 |
| Bedrooms | 4 |
| Year Built | 2000 |
| Lot Size | 1.72 |
| Garage Size | 1 |

Output

```
Predicted House Price:
$XXX,XXX.XX
```

---

## Future Improvements

- Compare multiple regression algorithms.
- Perform feature engineering.
- Add cross-validation.
- Implement GridSearchCV for hyperparameter tuning.
- Deploy the application on Streamlit Community Cloud.
- Add model performance visualizations.
- Improve the UI with charts and analytics.

---

## License

This project is licensed under the MIT License.
