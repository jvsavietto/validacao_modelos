from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt  
from sklearn.metrics import plot_confusion_matrix

X, y = load_breast_cancer(return_X_y=True)

# Hold-out
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print('Score hold-out:', model.score(X_test, y_test))
plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, display_labels=['malignant', 'benign'])

# validação cruzada k-fold

model = RandomForestClassifier(random_state=42)
scores = cross_val_score(model, X, y, cv=10)

print('Scores 10-fold cross-validation:', scores)
print('Média 10-fold cross-validation:', scores.mean() )
print('Desvio padrão 10-fold cross-validation:', scores.std())

