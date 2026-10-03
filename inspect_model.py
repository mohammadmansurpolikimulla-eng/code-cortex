import tensorflow as tf
import joblib
import json

print("Loading model...")
model = tf.keras.models.load_model('op/ann_24h.keras')
model.summary()

print("\nLoading scalers...")
scalers = joblib.load('op/scalers.joblib')
print(scalers.keys())

print("\nDone.")
