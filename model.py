import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load and prepare the dataset
data = pd.read_csv('ev_charging_patterns.csv')  # Modify with the appropriate path

energy_trungvi = data['Energy Consumed (kWh)'].median()
data['Energy Consumed (kWh)'].fillna(energy_trungvi, inplace=True)
chargingrate_trungvi = data['Charging Rate (kW)'].median()
data['Charging Rate (kW)'].fillna(chargingrate_trungvi, inplace=True)
distance_trungvi = data['Distance Driven (since last charge) (km)'].median()
data['Distance Driven (since last charge) (km)'].fillna(distance_trungvi, inplace=True)

data['Charging Station Location'] = data['Charging Station Location'].map({"Los Angeles" : 1, "San Francisco" : 2, "Houston " : 3, "New York" : 4, "Chicago  " : 5})
data['Vehicle Model'] = data['Vehicle Model'].map({"Tesla Model 3" : 1, "Hyundai Kona" : 2, "Nissan Leaf" : 3, "BMW i3" : 4, "Chevy Bolt" : 5})
data['Time of Day'] = data['Time of Day'].map({"Evening" : 1, "Morning" : 2, "Night" : 3, "Afternoon" : 4})
data['Day of Week'] = data['Day of Week'].map({"Monday" : 1, "Tuesday" : 2, "Wednesday" : 3, "Thursday" : 4, "Friday" : 5, "Saturday" : 6, "Sunday" : 7})
data['Charger Type'] = data['Charger Type'].map({"Level 1" : 1, "Level 2" : 2, "DC Fast Charger" : 3})
data['User Type'] = data['User Type'].map({"Commuter" : 1, "Long-Distance Traveler" : 2, "Casual Driver" : 3})

inputs = data[['Energy Consumed (kWh)', 'Charging Duration (hours)', 'Charging Rate (kW)', 
               'Charging Station Location', 'Charger Type', 'Time of Day', 'Day of Week']]
output = data['Charging Cost (USD)']

inputs_encoded = pd.get_dummies(inputs, columns=['Charging Station Location', 'Charger Type', 'Time of Day', 'Day of Week'])

X_train, X_test, y_train, y_test = train_test_split(inputs_encoded, output, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# --- 1. Linear Regression in PyTorch ---
# class LinearRegressionModel(nn.Module):
#     def __init__(self, input_size):
#         super(LinearRegressionModel, self).__init__()
#         self.linear = nn.Linear(input_size, 1)
    
#     def forward(self, x):
#         return self.linear(x)

# # Initialize and train Linear Regression model
# linear_model = LinearRegressionModel(X_train_tensor.shape[1])
# criterion = nn.MSELoss()
# optimizer = optim.SGD(linear_model.parameters(), lr=0.01)

# # Training loop
# epochs = 100
# for epoch in range(epochs):
#     linear_model.train()
#     optimizer.zero_grad()
    
#     # Forward pass
#     outputs = linear_model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)
    
#     # Backward pass
#     loss.backward()
#     optimizer.step()
    
#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# # Testing Linear Regression
# linear_model.eval()
# with torch.no_grad():
#     y_pred_lr = linear_model(X_test_tensor).numpy().flatten()

# rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
# r2_lr = r2_score(y_test, y_pred_lr)
# print(f"Linear Regression - RMSE: {rmse_lr:.4f}, R2 Score: {r2_lr:.4f}")

# --- 2. Random Forest Regressor ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest Regressor - RMSE: {rmse_rf:.4f}, R2 Score: {r2_rf:.4f}")

# --- 3. Gradient Boosting Regressor ---
# gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# gb_model.fit(X_train, y_train)
# y_pred_gb = gb_model.predict(X_test)

# rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
# r2_gb = r2_score(y_test, y_pred_gb)
# print(f"Gradient Boosting Regressor - RMSE: {rmse_gb:.4f}, R2 Score: {r2_gb:.4f}")

# --- 4. XGBoost Regressor ---
# xgb_model = XGBRegressor(n_estimators=100, random_state=42)
# xgb_model.fit(X_train, y_train)
# y_pred_xgb = xgb_model.predict(X_test)

# rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
# r2_xgb = r2_score(y_test, y_pred_xgb)
# print(f"XGBoost Regressor - RMSE: {rmse_xgb:.4f}, R2 Score: {r2_xgb:.4f}")
