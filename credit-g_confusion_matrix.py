# IMPORTS
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  
from sklearn.metrics import plot_confusion_matrix

# PRÉ-PROCESSAMENTO
dataset = pd.read_csv('credit-g.csv')
dataset_dummies = pd.get_dummies(dataset, columns=['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment', 
                                'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans', 'housing',
                                'job', 'own_telephone', 'foreign_worker'], drop_first=True)

labelencoder = LabelEncoder()
dataset_dummies['class'] = labelencoder.fit_transform(dataset_dummies['class'])
X = dataset_dummies.drop(axis=1, columns=['class'])
y = dataset_dummies[['class']]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = y.values.ravel()

# TREINAMENTO DO MODELO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(25, 25), max_iter=200, random_state=42)
model.fit(X_train, y_train)
plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, display_labels=['bad', 'good'])