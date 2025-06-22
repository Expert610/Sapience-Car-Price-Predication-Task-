import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df = pd.read_csv('car data.csv')
df.head()

# df.dropna(inplace=True)
# print(df.shape)

df.drop(['Car_Name'],axis=1,inplace=True)
df['Car_Age'] = 2025 - df['Year']

df.drop(['Year'],axis=1,inplace=True)
df.head()

# print(df['Fuel_Type'].value_counts())
# print(df['Selling_type'].value_counts())
# print(df['Transmission'].value_counts())
# print(df['Owner'].value_counts())

df = pd.get_dummies(df,columns=['Fuel_Type','Selling_type','Transmission'],drop_first=True)

X = df.drop(['Selling_Price'],axis=1)
Y = df['Selling_Price']

X_train ,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

model = RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,Y_train)

y_predict = model.predict(X_test)

r2 = r2_score(Y_test,y_predict)
mse = mean_squared_error(Y_test, y_predict)

# Show the results
print(f"R² Score: {r2:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

y_test_sorted, y_pred_sorted = zip(*sorted(zip(Y_test, y_predict)))

sns.set_style("whitegrid")
plt.figure(figsize=(10,6))

sns.lineplot(data=y_test_sorted, label="Actual Price" , color="green" ,linestyle="--" ,linewidth=2,markers="o")
sns.lineplot(data=y_pred_sorted,label="Predicted Price", color="blue",linewidth=2, linestyle="-",markers="s")
plt.title('Actual vs Predicted Car Prices (Line Chart)')

plt.title("Actual vs Predicted Car Prices")
plt.xlabel("Sample Index")
plt.ylabel("Price (in Lakh)")
plt.legend()
plt.tight_layout()
plt.show()


# plt.figure(figsize=(10, 6))
# plt.plot(y_test_sorted, label='Actual Price', color='green', linewidth=2)
# plt.plot(y_pred_sorted, label='Predicted Price', color='orange', linestyle='--', linewidth=2)
# plt.title('Actual vs Predicted Car Prices (Line Chart)')
# plt.xlabel('Sample Index')
# plt.ylabel('Price (in Lakh)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



n = 20
indices = np.arange(n)
bar_width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(indices, Y_test[:n], width=bar_width, label='Actual Price', color='steelblue')
plt.bar(indices + bar_width, y_predict[:n], width=bar_width, label='Predicted Price', color='coral')
plt.xlabel('Sample Index')
plt.ylabel('Price (in Lakh)')
plt.title('Actual vs Predicted Car Prices (Bar Chart)')
plt.xticks(indices + bar_width / 2, labels=[str(i) for i in range(n)])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
