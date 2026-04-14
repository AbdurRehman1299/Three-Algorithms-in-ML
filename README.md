# 🧠 Image Classification from Scratch: KNN, Naive Bayes, & Decision Tree

A full-stack web application that allows users to train, evaluate, and test classical machine learning algorithms on a custom image dataset. The core machine learning algorithms were built **entirely from scratch** using standard Python and NumPy—no `scikit-learn` or high-level ML libraries were used!

## ✨ Features

- **Algorithms Built from Scratch:** Includes custom implementations of K-Nearest Neighbors, Gaussian/Bernoulli Naive Bayes, and a Decision Tree classifier.
- **Interactive Dashboard:** A clean HTML/JS front-end to select models, trigger training, and view performance metrics.
- **Live Evaluation Metrics:** Automatically calculates Accuracy, Precision, Recall, F1-Score, and generates a Confusion Matrix upon training.
- **Custom Image Upload:** Train a model and instantly test it by uploading your own image to see its prediction in real-time.
- **Optimized for Speed:** Includes hyperparameter tuning and a Random-Forest-style square-root feature selection hack to allow the Decision Tree to process image pixels efficiently.

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Machine Learning / Math:** NumPy, Pillow (PIL), Collections
- **Frontend:** HTML5, CSS3, Vanilla JavaScript (Fetch API)

## 📁 Project Structure

```text
├── dataset/
│   ├── training_set/
│   │   ├── cats/      # Put training cat images here
│   │   └── dogs/      # Put training dog images here
│   └── test_set/
│       ├── cats/      # Put testing cat images here
│       └── dogs/      # Put testing dog images here
├── templates/
│   └── index.html     # The web dashboard frontend
├── app.py             # Flask server & image processing pipeline
├── ml_algorithms.py   # The from-scratch math & ML logic
└── README.md
```

# 🚀 Installation & Setup

### 1. **Clone the repository:**
```bash
git clone https://github.com/AbdurRehman1299/Three-Algorithms-in-ML.git
```

```bash
cd Three-Algorithms-in-ML
```
### 2. **Create the virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. **Install the required Python packages:**
```bash
pip install -r requirements.txt
```

### 4. **Run the Flask Application:**
```bash
python app.py
```

### 5. **Open the web application:**
Open your browser and navigate to ```http://127.0.0.1:5000/```

# ⚙️ How It Works (The ML Pipeline)

Because these are classical ML algorithms handling image data, the data goes through a strict preprocessing pipeline before hitting the math:

#### 1. **Grayscale Conversion:** Images are converted to black and white ('L' mode) to reduce complexity.

#### 2. **Resizing:** Images are scaled down to 32x32 pixels.

#### 3. **Flattening:** The 2D image matrix is flattened into a 1D array of 1,024 features.

#### 4. **Normalization:** Pixel brightness values are divided by 255.0 to map them between 0 and 1.

### The Algorithms

- **K-Nearest Neighbors:** Uses Euclidean distance to compare the 1,024 pixels of a new image against every image in memory. Tuned to k=7.

- **Naive Bayes:** Binarizes pixels based on a brightness threshold (> 0.6) and calculates log-probabilities based on dataset frequencies.

- **Decision Tree:** Uses Information Gain (Entropy) to split data. To prevent extreme training times on 1,024 features, it uses np.sqrt() feature subsampling at each node split, tuned to a max_depth of 15.

# ⚠️ Known Limitations (Classical ML vs. Images)

This project exists as an educational exercise in writing fundamental algorithms from scratch. Classical algorithms (KNN, Trees) view images as a flat list of brightness numbers, completely ignoring shapes, outlines, and spatial context.
Therefore, accuracy generally peaks around 55-65% on complex images like cats and dogs. For production-grade image recognition, deep learning (CNNs) is recommended.