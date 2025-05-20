import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

house_df = pd.read_csv("HouseData.csv")
house_df.columns = house_df.columns.str.strip()

kadikoy_df = house_df[house_df['district'].str.lower() == 'kadikoy'].copy()

kadikoy_df['price'] = kadikoy_df['price'].str.replace("TL", "").str.replace(",", "").str.strip()
kadikoy_df['price'] = pd.to_numeric(kadikoy_df['price'], errors='coerce')
kadikoy_df['GrossSquareMeters'] = kadikoy_df['GrossSquareMeters'].str.replace("m2", "").str.strip()
kadikoy_df['GrossSquareMeters'] = pd.to_numeric(kadikoy_df['GrossSquareMeters'], errors='coerce')
kadikoy_df['NetSquareMeters'] = kadikoy_df['NetSquareMeters'].str.replace("m2", "").str.strip()
kadikoy_df['NetSquareMeters'] = pd.to_numeric(kadikoy_df['NetSquareMeters'], errors='coerce')


def parse_age(age):
    if "Yeni" in str(age):
        return 0
    elif "5-10" in str(age):
        return 7
    elif "21" in str(age):
        return 25
    else:
        try:
            return int(age)
        except:
            return None
kadikoy_df['BuildingAge'] = kadikoy_df['BuildingAge'].apply(parse_age)

def parse_rooms(room):
    try:
        parts = str(room).split('+')
        return sum([int(p) for p in parts if p.isdigit()])
    except:
        return None
kadikoy_df['NumberOfRooms'] = kadikoy_df['NumberOfRooms'].apply(parse_rooms)

kadikoy_df['NumberOfBathrooms'] = pd.to_numeric(kadikoy_df['NumberOfBathrooms'], errors='coerce')

kadikoy_df['NumberOfBalconies'] = pd.to_numeric(kadikoy_df['NumberOfBalconies'], errors='coerce')

kadikoy_df['NumberFloorsofBuilding'] = pd.to_numeric(kadikoy_df['NumberFloorsofBuilding'], errors='coerce')

features = [
    'GrossSquareMeters', 'NetSquareMeters', 'BuildingAge', 'NumberOfRooms',
    'NumberOfBathrooms', 'NumberOfBalconies', 'NumberFloorsofBuilding',
    'FloorLocation', 'HeatingType', 'CreditEligibility'
]

kadikoy_df = kadikoy_df.dropna(subset=['price'] + features)

for col in ['price', 'GrossSquareMeters', 'NetSquareMeters', 'BuildingAge', 'NumberOfRooms', 'NumberOfBathrooms', 'NumberOfBalconies', 'NumberFloorsofBuilding']:
    lower = kadikoy_df[col].quantile(0.01)
    upper = kadikoy_df[col].quantile(0.99)
    kadikoy_df = kadikoy_df[(kadikoy_df[col] >= lower) & (kadikoy_df[col] <= upper)]

kadikoy_df['log_price'] = np.log1p(kadikoy_df['price'])

categorical = ['FloorLocation', 'HeatingType', 'CreditEligibility']
X = kadikoy_df[features]
X = pd.get_dummies(X, columns=categorical, drop_first=True)
y = kadikoy_df['log_price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
rf_model = grid_search.best_estimator_
y_pred_rf = rf_model.predict(X_test_scaled)

xgb_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

y_test_exp = np.expm1(y_test)
y_pred_lr_exp = np.expm1(y_pred_lr)
y_pred_rf_exp = np.expm1(y_pred_rf)
y_pred_xgb_exp = np.expm1(y_pred_xgb)

# Feature importance from Random Forest
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

def draw_graph(y_test, y_pred, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.xlabel("Actual Price", fontsize=12)
    plt.ylabel("Predicted Price", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True)
    st.pyplot(plt)

def draw_feature_importance():
    plt.figure(figsize=(10, 6))
    feature_importances.head(10).plot(kind='bar')
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.ylabel('Importance')
    plt.tight_layout()
    st.pyplot(plt)

def draw_performance_metrics():
    models = ['Linear Regression', 'Random Forest', 'XGBoost']
    mse_values = [
        mean_squared_error(y_test_exp, y_pred_lr_exp),
        mean_squared_error(y_test_exp, y_pred_rf_exp),
        mean_squared_error(y_test_exp, y_pred_xgb_exp)
    ]
    r2_values = [
        r2_score(y_test_exp, y_pred_lr_exp),
        r2_score(y_test_exp, y_pred_rf_exp),
        r2_score(y_test_exp, y_pred_xgb_exp)
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.bar(models, mse_values, color=['blue', 'green', 'red'])
    ax1.set_title('Test Loss (MSE)')
    ax1.set_ylabel('MSE')
    ax1.set_xticklabels(models, rotation=45)
    
    ax2.bar(models, r2_values, color=['blue', 'green', 'red'])
    ax2.set_title('Test Accuracy (R²)')
    ax2.set_ylabel('R²')
    ax2.set_xticklabels(models, rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

# Streamlit Interface
st.title("Kadikoy House Price Prediction (Advanced)")

st.sidebar.header("Input Features")
input_gross = st.sidebar.number_input("Gross Square Meters:", min_value=30, max_value=400, value=100)
input_net = st.sidebar.number_input("Net Square Meters:", min_value=20, max_value=350, value=80)
input_age = st.sidebar.number_input("Building Age:", min_value=0, max_value=100, value=10)
input_rooms = st.sidebar.number_input("Number of Rooms:", min_value=1, max_value=10, value=3)
input_bathrooms = st.sidebar.number_input("Number of Bathrooms:", min_value=1, max_value=5, value=1)
input_balconies = st.sidebar.number_input("Number of Balconies:", min_value=0, max_value=5, value=1)
input_floors = st.sidebar.number_input("Number of Floors in Building:", min_value=1, max_value=50, value=5)
input_floor = st.sidebar.selectbox("Floor Location:", sorted(kadikoy_df['FloorLocation'].dropna().unique()))
input_heating = st.sidebar.selectbox("Heating Type:", sorted(kadikoy_df['HeatingType'].dropna().unique()))
input_credit = st.sidebar.selectbox("Credit Eligibility:", sorted(kadikoy_df['CreditEligibility'].dropna().unique()))

# Prepare input for prediction
input_dict = {
    'GrossSquareMeters': input_gross,
    'NetSquareMeters': input_net,
    'BuildingAge': input_age,
    'NumberOfRooms': input_rooms,
    'NumberOfBathrooms': input_bathrooms,
    'NumberOfBalconies': input_balconies,
    'NumberFloorsofBuilding': input_floors,
    'FloorLocation': input_floor,
    'HeatingType': input_heating,
    'CreditEligibility': input_credit
}
input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df, columns=categorical, drop_first=True)
input_df = input_df.reindex(columns=X.columns, fill_value=0)
input_df_scaled = scaler.transform(input_df)

st.markdown("---")
if st.button("Predict Price (Linear Regression)"):
    pred_lr = lr_model.predict(input_df_scaled)[0]
    pred_lr_exp = np.expm1(pred_lr)
    st.success(f"Estimated Sale Price (Linear Regression): {int(pred_lr_exp):,} TL")
if st.button("Predict Price (Random Forest)"):
    pred_rf = rf_model.predict(input_df_scaled)[0]
    pred_rf_exp = np.expm1(pred_rf)
    st.success(f"Estimated Sale Price (Random Forest): {int(pred_rf_exp):,} TL")
if st.button("Predict Price (XGBoost)"):
    pred_xgb = xgb_model.predict(input_df_scaled)[0]
    pred_xgb_exp = np.expm1(pred_xgb)
    st.success(f"Estimated Sale Price (XGBoost): {int(pred_xgb_exp):,} TL")

st.markdown("---")
st.subheader("Model Performance")
st.write(f"Linear Regression - MSE: {mean_squared_error(y_test_exp, y_pred_lr_exp):,.0f}, R²: {r2_score(y_test_exp, y_pred_lr_exp):.2f}")
st.write(f"Random Forest - MSE: {mean_squared_error(y_test_exp, y_pred_rf_exp):,.0f}, R²: {r2_score(y_test_exp, y_pred_rf_exp):.2f}")
st.write(f"XGBoost - MSE: {mean_squared_error(y_test_exp, y_pred_xgb_exp):,.0f}, R²: {r2_score(y_test_exp, y_pred_xgb_exp):.2f}")

with st.expander("Show Prediction Graphs"):
    draw_graph(y_test_exp, y_pred_lr_exp, "Actual vs Predicted Price (Linear Regression)")
    draw_graph(y_test_exp, y_pred_rf_exp, "Actual vs Predicted Price (Random Forest)")
    draw_graph(y_test_exp, y_pred_xgb_exp, "Actual vs Predicted Price (XGBoost)")

with st.expander("Show Feature Importances (Random Forest)"):
    draw_feature_importance()

with st.expander("Model Performance Visualization"):
    draw_performance_metrics()
