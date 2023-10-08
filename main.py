import pandas  as pd
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
def train_model():
    data = pd.read_csv('src/predictive_maintenance.csv')

    new_labels = ['uid', 'prod_id', 'prod_type', 'air_temp', 'process_temp', 'rot_speed', 'torque', 'tool_wear', 'target', 'failure_type']

    data.columns = new_labels
    X = data[['torque','process_temp','tool_wear', "air_temp", "rot_speed"]]
    y = data['target']


    # Dividindo os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    # Visualizando o shape dos dados

    sm= SMOTE(random_state=2)
    model = Sequential([
        # Input layer
        Dense(units=5, input_shape=(5,), activation='relu', name='input_layer', ),
        # Hidden Layer -> ReLU
        Dense(units=4, activation='relu'),
        # Hidden Layer -> Sigmoid
        Dense(units=4, activation='sigmoid'),
        # Output layer -> Prone to failure or not prone to failure
        Dense(units=2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

    # model.summary()

    model.fit(X_train, y_train, epochs=150, batch_size=16, validation_split=0.2, verbose=2)

    loss, acc = model.evaluate(X_test, y_test, verbose=2)

    print(f"Accuracy: {acc*100:.2f}%, Loss: {loss*100:.2f}%")

    model.save('src/model.h5')
    return model

def use_model():
    """
    Function to feed in real time data to the model. Data is found at output folder
    """
    try:
        model = keras.models.load_model('src/model.h5')
    except OSError:
        model = train_model()
        model = keras.models.load_model('src/model.h5')
    finally:
        data = pd.read_csv('output/realtime_data.csv')
        data = data[['torque','process_temp','tool_wear', "air_temp", "rot_speed"]]
        predictions = model.predict(data)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_class = np.argmax(predictions, axis=1)
        
        # Print predictions with indication of failure
        for i, prediction in enumerate(predicted_class):
            if prediction == 1:
                print(f"Machine {i + 1}: Predicted Failure")
            else:
                continue


if __name__ == "__main__":
    use_model()