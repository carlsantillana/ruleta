import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def monte_carlo_simulation(numbers, simulations=10000):
    outcomes = np.zeros(37)
    for _ in range(simulations):
        outcome = np.random.choice(numbers)
        outcomes[outcome] += 1
    probabilities = outcomes / simulations
    return np.argmax(probabilities)

def get_recommendation(numbers):
    if not numbers:
        return "No data available"
    recommendation = monte_carlo_simulation(numbers)
    return recommendation

def train_model(numbers):
    X = []
    y = []
    for i in range(1, len(numbers)):
        X.append(numbers[:i])
        y.append(numbers[i])
    
    max_len = max(len(x) for x in X)
    X = np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in X])
    y = np.array(y)
    
    # Asegurarnos de que todas las clases estén representadas en el entrenamiento
    y_unique, y_counts = np.unique(y, return_counts=True)
    missing_classes = set(range(37)) - set(y_unique)
    if missing_classes:
        for cls in missing_classes:
            X = np.vstack([X, np.zeros((max_len,))])
            y = np.append(y, cls)
    
    model = LogisticRegression(max_iter=10000)
    model.fit(X, y)
    
    with open('roulette_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model_meta.pkl', 'wb') as f:
        pickle.dump({'max_len': max_len}, f)

def load_model():
    if not os.path.exists('roulette_model.pkl') or not os.path.exists('model_meta.pkl'):
        return None, None
    with open('roulette_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    return model, meta['max_len']

def reset_model():
    files = ['roulette_model.pkl', 'model_meta.pkl', 'color_model.pkl', 'parity_model.pkl', 'range_model.pkl']
    for file in files:
        if os.path.exists(file):
            os.remove(file)

def predict_next_numbers(model, numbers, max_len, num_predictions=10):
    if max_len is None:
        return ["Model not trained yet"]
    if len(numbers) > max_len:
        numbers = numbers[-max_len:]
    X = np.array([np.pad(numbers, (0, max_len - len(numbers)), 'constant')])
    probabilities = model.predict_proba(X)[0]
    recommended_numbers = np.argsort(probabilities)[-num_predictions:][::-1]
    return sorted(recommended_numbers)

def train_color_model(numbers):
    red_numbers = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
    colors = ['red' if num in red_numbers else 'black' for num in numbers]
    X = []
    y = []
    for i in range(1, len(numbers)):
        X.append(numbers[:i])
        y.append(colors[i])
    
    max_len = max(len(x) for x in X)
    X = np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in X])
    y = np.array(y)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    color_model = LogisticRegression(max_iter=10000)
    color_model.fit(X_train, y_train)
    
    with open('color_model.pkl', 'wb') as f:
        pickle.dump((color_model, label_encoder), f)

def load_color_model():
    if not os.path.exists('color_model.pkl'):
        return None, None
    with open('color_model.pkl', 'rb') as f:
        color_model, label_encoder = pickle.load(f)
    return color_model, label_encoder

def predict_color(model, numbers, max_len, label_encoder):
    if max_len is None or model is None:
        return "Model not trained yet"
    if len(numbers) > max_len:
        numbers = numbers[-max_len:]
    X = np.array([np.pad(numbers, (0, max_len - len(numbers)), 'constant')])
    color_probabilities = model.predict_proba(X)[0]
    predicted_color_index = np.argmax(color_probabilities)
    predicted_color = label_encoder.inverse_transform([predicted_color_index])[0]
    return predicted_color

def train_parity_model(numbers):
    parities = ['even' if num % 2 == 0 else 'odd' for num in numbers]
    X = []
    y = []
    for i in range(1, len(numbers)):
        X.append(numbers[:i])
        y.append(parities[i])
    
    max_len = max(len(x) for x in X)
    X = np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in X])
    y = np.array(y)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    parity_model = LogisticRegression(max_iter=10000)
    parity_model.fit(X_train, y_train)
    
    with open('parity_model.pkl', 'wb') as f:
        pickle.dump((parity_model, label_encoder), f)

def load_parity_model():
    if not os.path.exists('parity_model.pkl'):
        return None, None
    with open('parity_model.pkl', 'rb') as f:
        parity_model, label_encoder = pickle.load(f)
    return parity_model, label_encoder

def predict_parity(model, numbers, max_len, label_encoder):
    if max_len is None or model is None:
        return "Model not trained yet"
    if len(numbers) > max_len:
        numbers = numbers[-max_len:]
    X = np.array([np.pad(numbers, (0, max_len - len(numbers)), 'constant')])
    parity_probabilities = model.predict_proba(X)[0]
    predicted_parity_index = np.argmax(parity_probabilities)
    predicted_parity = label_encoder.inverse_transform([predicted_parity_index])[0]
    return predicted_parity

def train_range_model(numbers):
    ranges = ['1-12' if 1 <= num <= 12 else '13-24' if 13 <= num <= 24 else '25-36' for num in numbers]
    X = []
    y = []
    for i in range(1, len(numbers)):
        X.append(numbers[:i])
        y.append(ranges[i])
    
    max_len = max(len(x) for x in X)
    X = np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in X])
    y = np.array(y)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    range_model = LogisticRegression(max_iter=10000)
    range_model.fit(X_train, y_train)
    
    with open('range_model.pkl', 'wb') as f:
        pickle.dump((range_model, label_encoder), f)

def load_range_model():
    if not os.path.exists('range_model.pkl'):
        return None, None
    with open('range_model.pkl', 'rb') as f:
        range_model, label_encoder = pickle.load(f)
    return range_model, label_encoder

def predict_range(model, numbers, max_len, label_encoder, num_predictions=2):
    if max_len is None or model is None:
        return "Model not trained yet"
    if len(numbers) > max_len:
        numbers = numbers[-max_len:]
    X = np.array([np.pad(numbers, (0, max_len - len(numbers)), 'constant')])
    range_probabilities = model.predict_proba(X)[0]
    predicted_range_indices = np.argsort(range_probabilities)[-num_predictions:][::-1]
    predicted_ranges = label_encoder.inverse_transform(predicted_range_indices)
    return sorted(predicted_ranges)
