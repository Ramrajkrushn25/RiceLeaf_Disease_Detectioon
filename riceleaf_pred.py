from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ✅ Load your trained model
model = load_model("rice_leaf_model.keras")

# ✅ Define class labels (change based on your model)
class_labels = ["Bacterial leaf blight", "Brown spot", "Leaf smut", "Healthy"]

# ✅ Preprocess uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# ✅ Main route
@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    img_url = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            # Generate a unique filename to avoid overwrite
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Create upload folder if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            # Save file
            file.save(filepath)

            # Predict
            img = preprocess_image(filepath)
            result = model.predict(img)
            predicted_class = np.argmax(result, axis=1)[0]
            label = class_labels[predicted_class] if predicted_class < len(class_labels) else "Unknown"

            prediction = label
            img_url = filepath  # this will be used in HTML

    return render_template("result.html", prediction=prediction, img_url=img_url)

if __name__ == "__main__":
    app.run(debug=True)
