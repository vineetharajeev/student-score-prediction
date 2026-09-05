import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 

data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Scores": [35, 40, 45, 50, 55, 60, 65, 70, 80, 85]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Scores"]

model = LinearRegression()
model.fit(X, y)

hours = float(input("Enter study hours: "))

prediction = model.predict([[hours]])

print("\nStudy Hours:", hours)
print("Predicted Score:", round(prediction[0], 2))

plt.scatter(df["Hours"], df["Scores"])
plt.plot(df["Hours"], model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Study Hours vs Score")
plt.show()