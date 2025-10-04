import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('cardio_train.csv', sep=';')
df.drop('id', axis=1, inplace=True)
print("Missing values:\n", df.isnull().sum())

df['age'] = df['age'] // 365  # convert age from days to years

sns.barplot(y='cardio', x='age',hue='age', data=df, palette='coolwarm', legend=False)
plt.title("Cardiovascular Disease Percentage")
plt.show()

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

X = df.drop('cardio', axis=1)
y = df['cardio']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")

final_model = SVC()
final_model.fit(X_train, y_train)
final_preds = final_model.predict(X_test)

print("\nFinal Model Evaluation (SVM):")
print(confusion_matrix(y_test, final_preds))
print(classification_report(y_test, final_preds))