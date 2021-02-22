from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv("dataset/car.data")

X = df[["buying", "doors", "persons", "lug_boot", "safety"]].values
enc = OrdinalEncoder()
enc.fit(X)
X = enc.transform(X)

Y = df[["car_value"]]
le = LabelEncoder()
Y = le.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)

model = KNeighborsClassifier(n_neighbors=17)
model.fit(X_train, y_train)
prediction = model.predict(X_test)

print(accuracy_score(y_test, prediction))