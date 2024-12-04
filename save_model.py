import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib

data = pd.read_csv('ev_charging_patterns.csv')  

energy_trungvi = data['Energy Consumed (kWh)'].median()
data['Energy Consumed (kWh)'] = data['Energy Consumed (kWh)'].fillna(energy_trungvi)
chargingrate_trungvi = data['Charging Rate (kW)'].median()
data['Charging Rate (kW)'] = data['Charging Rate (kW)'].fillna(chargingrate_trungvi)
distance_trungvi = data['Distance Driven (since last charge) (km)'].median()
data['Distance Driven (since last charge) (km)'] = data['Distance Driven (since last charge) (km)'].fillna(distance_trungvi)

inputs = data[['Battery Capacity (kWh)', 'Charging Duration (hours)', 'Charging Rate (kW)', 
                 'Distance Driven (since last charge) (km)', 'Temperature (°C)', 'Vehicle Age (years)']]
output = data['Energy Consumed (kWh)']


scaler = StandardScaler()
X = scaler.fit_transform(inputs)
X_train, X_test, y_train, y_test = train_test_split(X, output, test_size=0.2, random_state=42)

#Linear model
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

#Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#XGB model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

#DNN model
import torch
import torch.nn as nn
import torch.optim as optim

class EnergyDNN(nn.Module):
    def __init__(self, input_size):
        super(EnergyDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

model_nn = EnergyDNN(X_train_tensor.shape[1])
optimizer = optim.Adam(model_nn.parameters(), lr=0.001)
criterion = nn.MSELoss()
epochs = 1000
for epoch in range(epochs):
    model_nn.train()
    optimizer.zero_grad()
    outputs = model_nn(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()


joblib.dump(model_linear, 'linear_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')
torch.save(model_nn.state_dict(), 'energy_dnn_model.pth')
