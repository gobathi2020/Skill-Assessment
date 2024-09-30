# Skill-Assessment

## Program
```py
# Register Number: 212222080017
# Name: Gobathi P

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df = pd.read_csv('FuelConsumption.csv')

# Q1: Scatter plot between CYLINDERS and CO2EMISSIONS (green color)
plt.figure(figsize=(10, 6))
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emissions')
plt.title('Scatter Plot: Cylinders vs CO2 Emissions')
plt.grid(True)
plt.show()

# Q2: Scatter plot to compare CYLINDERS vs CO2EMISSIONS and ENGINESIZE vs CO2EMISSIONS
plt.figure(figsize=(10, 6))
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green', label='Cylinders vs CO2 Emissions')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='blue', label='Engine Size vs CO2 Emissions')
plt.xlabel('Cylinders / Engine Size')
plt.ylabel('CO2 Emissions')
plt.title('Scatter Plot: Cylinders & Engine Size vs CO2 Emissions')
plt.legend()
plt.grid(True)
plt.show()

# Q3: Scatter plot to compare CYLINDERS, ENGINESIZE, and FUELCONSUMPTION_COMB vs CO2EMISSIONS
plt.figure(figsize=(10, 6))
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green', label='Cylinders vs CO2 Emissions')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='blue', label='Engine Size vs CO2 Emissions')
plt.scatter(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], color='red', label='Fuel Consumption (Combined) vs CO2 Emissions')
plt.xlabel('Cylinders / Engine Size / Fuel Consumption (Combined)')
plt.ylabel('CO2 Emissions')
plt.title('Scatter Plot: Cylinders, Engine Size & Fuel Consumption vs CO2 Emissions')
plt.legend()
plt.grid(True)
plt.show()


# Q4: Train model with CYLINDERS as independent variable and CO2EMISSIONS as dependent variable
X_cylinders = df[['CYLINDERS']]
y = df['CO2EMISSIONS']
X_train, X_test, y_train, y_test = train_test_split(X_cylinders, y, test_size=0.2, random_state=0)
model_cylinders = LinearRegression()
model_cylinders.fit(X_train, y_train)
y_pred_cylinders = model_cylinders.predict(X_test)
print("R^2 score for CYLINDERS vs CO2EMISSIONS:", r2_score(y_test, y_pred_cylinders))


# Q5: Train model with FUELCONSUMPTION_COMB as independent variable and CO2EMISSIONS as dependent variable
X_fuel = df[['FUELCONSUMPTION_COMB']]
X_train, X_test, y_train, y_test = train_test_split(X_fuel, y, test_size=0.2, random_state=0)
model_fuel = LinearRegression()
model_fuel.fit(X_train, y_train)
y_pred_fuel = model_fuel.predict(X_test)
print("R^2 score for FUELCONSUMPTION_COMB vs CO2EMISSIONS:", r2_score(y_test, y_pred_fuel))

# Q6: Train models on different train-test ratios and note their accuracies
ratios = [0.2, 0.3, 0.4]
for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X_fuel, y, test_size=ratio, random_state=0)
    model_fuel.fit(X_train, y_train)
    y_pred_fuel = model_fuel.predict(X_test)
    print(f"R^2 score for FUELCONSUMPTION_COMB vs CO2EMISSIONS with test size {ratio}: {r2_score(y_test, y_pred_fuel)}")


```

## Output

### Q1. Create a scatter plot between cylinder vs Co2Emission (green color)
![image](https://github.com/user-attachments/assets/f4a8b1fc-91a2-4018-8900-443fec5ce104)

### Q2. Using scatter plot compare data   cylinder vs Co2Emission and Enginesize Vs Co2Emission using different colors
![image](https://github.com/user-attachments/assets/efe8c33d-6ca2-4a15-8ba3-c8c9fd47db47)

### Q3. Using scatter plot compare data   cylinder vs Co2Emission and Enginesize Vs Co2Emission and FuelConsumption_comb Co2Emission using different colors
![image](https://github.com/user-attachments/assets/e4cfdadd-a183-4907-9062-97366fa84fc0)

### Q4. Train your model with independent variable as cylinder and dependent variable as Co2Emission
![image](https://github.com/user-attachments/assets/25b63b21-8608-4d20-90eb-4fac30c6a7a7)

### Q5. Train another model with independent variable as FuelConsumption_comb and dependent variable as Co2Emission
![image](https://github.com/user-attachments/assets/2d7e1a4a-ea25-42cd-930b-ac6c62a29e6a)

### Q6. Train your model on different train test ratio and train the models and note down their accuracies
![image](https://github.com/user-attachments/assets/66da423f-31d1-4ba1-a52f-7392edf5720c)
