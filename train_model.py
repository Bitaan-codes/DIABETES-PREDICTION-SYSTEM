import pandas as pd

df = pd.read_csv("diabetes.csv")
df

df_copy = df.copy()
df_copy

df_copy.isnull().sum()

df_copy.duplicated().sum()

x = df_copy.drop("Outcome", axis=1)
y = df_copy["Outcome"]

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
x,
y,
test_size = 0.20,
random_state = 42
)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print(x_train)

print(x_test)

x_train_df = pd.DataFrame(x_train,columns = x.columns)
x_train_df.head()

x_test_df = pd.DataFrame(x_test,columns = x.columns)
x_test_df.head()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 21)
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)
print(y_pred)

comparison = pd.DataFrame(
    {
        "Actual":y_test,
        "Predicted":y_pred
    }
)
print(comparison.head(10))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)

print("Accuracy :", accuracy)
print("Accuracy :", accuracy*100,"%")

print(y.value_counts())

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import precision_score
precision = precision_score(y_test,y_pred)
print("Precision :", precision)

from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)
print(recall)

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
print(f1)

new_patient = x.iloc[[5]]
new_patient = sc.transform(new_patient)
prediction = knn.predict(new_patient)
print(prediction)

import joblib
knn.fit(x_train,y_train)

joblib.dump(knn, "diabetes_prediction_model.pkl")


joblib.dump(sc, "diabetes_scaler.pkl")
