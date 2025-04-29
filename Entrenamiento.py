import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from imblearn.over_sampling import SMOTE


data = pd.read_csv('Hypertension-risk-model-main.csv')


data.dropna(axis=0, inplace=True)


X = data.drop('Risk', axis=1).values
y = data['Risk'].values


smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])


class_weight = {0: 1, 1: 1.2}


history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2, class_weight=class_weight)


y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.55).astype(int)

import joblib
joblib.dump(model,'IAhipertension.pkl')
model = joblib.load('IAhipertension.pkl')

joblib.dump(scaler,'IAscaler.pkl')
model1 = joblib.load('IAscaler.pkl')

print("Evaluación de la red neuronal:")
print(classification_report(y_test, y_pred, target_names=["No Hipertensión", "Hipertensión"]))