import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('ev_charging_patterns.csv')
# Giả sử data là DataFrame
# Lọc các cột đầu vào và đầu ra


# Xử lý giá trị thiếu
energy_trungvi = data['Energy Consumed (kWh)'].median()
data['Energy Consumed (kWh)'] = data['Energy Consumed (kWh)'].fillna(energy_trungvi)
chargingrate_trungvi = data['Charging Rate (kW)'].median()
data['Charging Rate (kW)'] = data['Charging Rate (kW)'].fillna(chargingrate_trungvi)
distance_trungvi = data['Distance Driven (since last charge) (km)'].median()
data['Distance Driven (since last charge) (km)'] = data['Distance Driven (since last charge) (km)'].fillna(distance_trungvi)

inputs = data[['Battery Capacity (kWh)', 'State of Charge (Start %)', 
               'Distance Driven (since last charge) (km)', 'Temperature (°C)']]
output = data['Energy Consumed (kWh)']

# Chia dữ liệu thành tập train-test
X_train, X_test, y_train, y_test = train_test_split(inputs, output, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Dự đoán
y_pred_lr = lin_reg.predict(X_test_scaled)

# Đánh giá
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression R² Score: {r2_lr:.4f}")


# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [10, 20, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Random Forest Model
# rf_model = RandomForestRegressor(random_state=42)
# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='r2', verbose=2)
# grid_search.fit(X_train_scaled, y_train)

# # Best Model
# best_rf_model = grid_search.best_estimator_

# # Dự đoán
# y_pred_rf = best_rf_model.predict(X_test_scaled)

# # Đánh giá
# r2_rf = r2_score(y_test, y_pred_rf)
# print(f"Optimized Random Forest R² Score: {r2_rf:.4f}")

from xgboost import XGBRegressor

# Define the XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Dự đoán
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Đánh giá
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"XGBoost R² Score: {r2_xgb:.4f}")

import torch
import torch.nn as nn
import torch.optim as optim

# Convert dữ liệu sang Tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define the DNN model
class EnergyDNN(nn.Module):
    def __init__(self, input_size):
        super(EnergyDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model
input_size = X_train_tensor.shape[1]
model = EnergyDNN(input_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Testing
model.eval()
with torch.no_grad():
    y_pred_nn = model(X_test_tensor).numpy().flatten()

# Đánh giá
r2_nn = r2_score(y_test, y_pred_nn)
print(f"DNN R² Score: {r2_nn:.4f}")

