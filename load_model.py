from model import create_model

model = create_model()

model.load_weights("emotion.weights.h5")

print("✅ Weights Loaded Successfully")