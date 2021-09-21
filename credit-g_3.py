# IMPORTS
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder 

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

model = MLPClassifier(hidden_layer_sizes=(25, 25), max_iter=200, random_state=42)
model.fit(X_train, y_train)
scores = cross_val_score(model, X, y, cv=10)
print(scores)
# OUTPUT: [0.78 0.68 0.68 0.72 0.72 0.7  0.73 0.72 0.74 0.68]