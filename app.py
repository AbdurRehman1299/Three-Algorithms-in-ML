import os
from flask import Flask, render_template, request, jsonify
from ml_algorithms import KNN, NaiveBayes, DecisionTree, accuracy, metrics, confusion_matrix
import numpy as np
from PIL import Image
from collections import Counter

app = Flask(__name__)
active_model = None
reverse_class_map = {}

def load_dataset(base_folder, class_mapping=None, image_size=(16, 16)):
    X = []
    y = []

    # Identify the class folders (cats, dogs)
    classes = sorted([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))])

    # If we are loading the training set, create the mapping {'cats': 0, 'dogs': 1}
    # If we are loading the testing set, use the existing mapping
    if class_mapping is None:
        class_mapping = {cls_name: idx for idx, cls_name in enumerate(classes)}

    for cls_name in classes:
        # Skip if the test folder has a random class not found in training
        if cls_name not in class_mapping:
            continue

        cls_folder = os.path.join(base_folder, cls_name)
        for img_name in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, img_name)
            try:
                # Open, convert to grayscale, resize, and flatten to 1D
                img = Image.open(img_path).convert('L').resize(image_size)
                img_array = np.array(img).flatten() / 255.0
                X.append(img_array)
                y.append(class_mapping[cls_name])
            except Exception as e:
                print(f'Warning: Skipped {img_name} - {e}')


    return np.array(X), np.array(y), class_mapping

# Load Images from Train and Test Folders
try:
    print('Loading Training Image...')
    X_train, y_train, class_map = load_dataset('dataset/training_set', image_size=(32, 32))
    reverse_class_map = {v: k for k, v in class_map.items()}
    distribution = Counter(y_train)
    print(f'Loaded {len(X_train)} training images. Classes: {class_map}')
    print(f'Class Distribution: Cats = {distribution.get(0, 0)} | Dogs = {distribution.get(1, 0)}')

    print('Loading Testing Images...')
    X_test, y_test, _ = load_dataset('dataset/test_set', class_mapping=class_map, image_size=(32, 32))
    print(f'Loaded {len(X_test)} testing images.')
except Exception as e:
    print(f'Error loading dataset: {e}')
    X_train = X_test = np.array([])
    y_train = y_test = np.array([])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    global active_model
    global X_train, X_test, y_train, y_test
    if len(X_train) == 0 or len(X_test) == 0:
        return jsonify({"error": "Data missing. Check your dataset/training_set and dataset/test_set folders."}), 400
    
    model_type = request.json.get('model')

    if model_type == 'knn':
        model = KNN(k=7)
    elif model_type == 'naive_bayes':
        model = NaiveBayes()
    elif model_type == 'decision_tree':
        model = DecisionTree(max_depth=15, min_samples_split=5)
    else:
        return jsonify({"error": f"Unknown model type: {model_type}"}), 400

    # Traing using the specific train folder
    model.fit(X_train, y_train)
    active_model = model

    # Predict using the specific test folder
    y_pred = model.predict(X_test)

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Calculate Metrics
    acc = accuracy(y_test, y_pred)
    precision, recall, f1 = metrics(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    return jsonify({
        "accuracy": round(acc * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": conf_matrix
    })

@app.route('/predict-image', methods=['POST'])
def predict_image():
    if active_model is None:
        return jsonify({"error": "Please click one of the 'Run' buttons to train a model first!"}), 400
    
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files['file']
    try:
        # Process the image exactly like we processed the training data
        img = Image.open(file).convert('L').resize((32, 32))
        img_array = np.array(img).flatten() / 255.0 

        # Make the prediction
        img_array_2d = np.array([img_array])
        pred_num = active_model.predict(img_array_2d)[0]

        print("\n--- NEW IMAGE UPLOADED ---")
        print(f"Model mathematically guessed number: {pred_num}")
        print(f"My dictionary is translating it using: {reverse_class_map}")
        
        # Translate the number back to the animal name
        pred_label = reverse_class_map.get(pred_num, "Unknown")

        return jsonify({"prediction": pred_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)