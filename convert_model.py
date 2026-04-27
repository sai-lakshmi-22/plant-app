from tensorflow.keras.models import load_model

model = load_model(
    "model/fixed_plant_model.h5",
    compile=False
)

model.save("model/plant_model.keras")

print("Saved successfully")
