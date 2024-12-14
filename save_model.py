import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from sklearn.svm import SVR

data = pd.read_csv('ev_data.csv')  

inputs = data[['Charging Time', 'Average Power', 'SOC Charged']]
output = data['Total Energy Delivered']


scaler = StandardScaler()
X = scaler.fit_transform(inputs)
X_train, X_test, y_train, y_test = train_test_split(X, output, test_size=0.2, random_state=42)

#Linear model
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

#Ridge model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

#Lasso model
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

#Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

#Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#SVM model
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(X_train, y_train)

#DNN model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class EnergyDNN(nn.Module):
    def __init__(self, input_size):
        super(EnergyDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)
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
joblib.dump(ridge_model, 'ridge_model.pkl')
joblib.dump(lasso_model,'lasso_model.pkl')
joblib.dump(gb_model,'gb_model.pkl')
joblib.dump(svr_model, 'svr_model.pkl')
torch.save(model_nn.state_dict(), 'energy_dnn_model.pth')
