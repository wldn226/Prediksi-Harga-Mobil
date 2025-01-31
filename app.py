import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

path_data = "car_price_dataset.csv"
data = pd.read_csv(path_data)

# Target harga langsung (bukan kategori harga)
y = data['Price']

numerical_columns = ['Year', 'Engine_Size', 'Mileage', 'Doors', 'Owner_Count']
categorical_columns = ['Brand', 'Model', 'Fuel_Type', 'Transmission']

X = data.drop(columns=['Price'])

# Preprocessing: One-Hot Encoding untuk kolom kategorikal, passthrough untuk numerik
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Transformasi data
X_encoded = preprocessor.fit_transform(X)

# Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Model regresi
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Support Vector Machine": SVR(),
}

# Train dan evaluasi setiap model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2 Score": r2_score(y_test, y_pred),
    }
# Convert results to DataFrame
results_df = pd.DataFrame(results).T


# Streamlit UI
st.title("Mobil Price Prediction & Model Evaluation")
st.subheader("Model Evaluation Results:")
st.dataframe(results_df)

# Form untuk pencarian mobil berdasarkan fitur
st.subheader("Find Car Price Prediction")

brand = st.selectbox("Select Car Brand", data['Brand'].unique())
model = st.selectbox("Select Car Model", data[data['Brand'] == brand]['Model'].unique())
year = st.number_input("Enter Car Year", min_value=1900, max_value=2025)
engine_size = st.number_input("Enter Engine Size", min_value=0.5, max_value=10.0)
mileage = st.number_input("Enter Mileage (km)", min_value=0, max_value=1000000)
doors = st.number_input("Enter Number of Doors", min_value=2, max_value=5)
owner_count = st.number_input("Enter Number of Owners", min_value=1, max_value=5)

if st.button("Predict Price"):
    user_data = pd.DataFrame({
        'Brand': [brand],
        'Model': [model],
        'Year': [year],
        'Engine_Size': [engine_size],
        'Mileage': [mileage],
        'Doors': [doors],
        'Owner_Count': [owner_count],
        'Fuel_Type': ['Petrol'], 
        'Transmission': ['Manual'],  
    })
    
    # Transformasi data pengguna
    user_data_encoded = preprocessor.transform(user_data)

    best_model = LinearRegression()  
    best_model.fit(X_train, y_train)
    
    # Prediksi harga
    predicted_price = best_model.predict(user_data_encoded)
    
    st.write(f"Predicted Car Price: ${predicted_price[0]:,.2f}")
