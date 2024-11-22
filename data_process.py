import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('ev_charging_patterns.csv')
energy_trungvi = df['Energy Consumed (kWh)'].median()
df['Energy Consumed (kWh)'].fillna(energy_trungvi, inplace=True)
chargingrate_trungvi = df['Charging Rate (kW)'].median()
df['Charging Rate (kW)'].fillna(chargingrate_trungvi, inplace=True)
distance_trungvi = df['Distance Driven (since last charge) (km)'].median()
df['Distance Driven (since last charge) (km)'].fillna(distance_trungvi, inplace=True)

df['Charging Station Location'] = df['Charging Station Location'].map({"Los Angeles" : 1, "San Francisco" : 2, "Houston " : 3, "New York" : 4, "Chicago  " : 5})
df['Vehicle Model'] = df['Vehicle Model'].map({"Tesla Model 3" : 1, "Hyundai Kona" : 2, "Nissan Leaf" : 3, "BMW i3" : 4, "Chevy Bolt" : 5})
df['Time of Day'] = df['Time of Day'].map({"Evening" : 1, "Morning" : 2, "Night" : 3, "Afternoon" : 4})
df['Day of Week'] = df['Day of Week'].map({"Monday" : 1, "Tuesday" : 2, "Wednesday" : 3, "Thursday" : 4, "Friday" : 5, "Saturday" : 6, "Sunday" : 7})
df['Charger Type'] = df['Charger Type'].map({"Level 1" : 1, "Level 2" : 2, "DC Fast Charger" : 3})
df['User Type'] = df['User Type'].map({"Commuter" : 1, "Long-Distance Traveler" : 2, "Casual Driver" : 3})

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

inputs = df[['Energy Consumed (kWh)', 'Charging Duration (hours)', 'Charging Rate (kW)', 
               'Charging Station Location', 'Charger Type', 'Time of Day', 'Day of Week']]
output = df['Charging Cost (USD)']

inputs_encoded = pd.get_dummies(inputs, columns=['Charging Station Location', 'Charger Type', 'Time of Day', 'Day of Week'])

X_train, X_test, y_train, y_test = train_test_split(inputs_encoded, output, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

class EVChargingCostModel(nn.Module):
    def __init__(self, input_size):
        super(EVChargingCostModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = X_train_tensor.shape[1]
model = EVChargingCostModel(input_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    
y_pred_np = y_pred.numpy().flatten()
y_test_np = y_test_tensor.numpy().flatten()

rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_np))
r2 = r2_score(y_test_np, y_pred_np)

print(f'RMSE: {rmse:.4f}')
print(f'R2 Score: {r2:.4f}')

