# Comprehensive AI Project: ML, DL, and NLP

This project is a practical demonstration of core skills across three major domains of Artificial Intelligence: Classical Machine Learning, Deep Learning, and Natural Language Processing.

---

## 📖 Project Overview

This repository contains the code for three distinct tasks:
1.  **Classical Machine Learning**: A Decision Tree classifier built with **Scikit-learn** to predict Iris flower species.
2.  **Deep Learning**: A Convolutional Neural Network (CNN) built with **TensorFlow/Keras** to recognize handwritten digits, achieving over 98% accuracy. This model is deployed in a web application using **Flask**.
3.  **Natural Language Processing**: An analysis of Amazon product reviews using **spaCy** to perform Named Entity Recognition (NER) and rule-based sentiment analysis.

---

## 🛠️ Technologies Used

-   **Backend**: Python
-   **Machine Learning**: Scikit-learn, Pandas
-   **Deep Learning**: TensorFlow (Keras)
-   **NLP**: spaCy
-   **Web Framework**: Flask
-   **Utilities**: NumPy, OpenCV, Pillow

---

## 📂 Project Structure

```
.
├── mnist_cnn_model.h5      # Trained deep learning model
├── app.py                  # Flask web application
├── cnn_mnist.py            # Script to train the CNN model
├── main.py                 # Script for the classical ML task (Iris)
├── nlp_spacy.py            # Script for the NLP task (Reviews)
├── requirements.txt        # Python dependencies
└── templates/
    └── index.html          # HTML/JS frontend for the Flask app
```

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create a Virtual Environment (Recommended)
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required libraries from the `requirements.txt` file. This also includes the spaCy language model.
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Task 1: Classical ML (Iris Classifier)
To run the Decision Tree training and evaluation script:
```bash
python main.py
```

### Task 2: NLP Analysis (Amazon Reviews)
To run the spaCy NER and sentiment analysis script:
```bash
python nlp_spacy.py
```

### Task 3: Deep Learning Web App (MNIST Recognizer)

**Step A: Train the Model (Optional)**
If the `mnist_cnn_model.h5` file is not present, you must first train the model by running:
```bash
python cnn_mnist.py
```
This will train the CNN and save the `mnist_cnn_model.h5` file.

**Step B: Launch the Flask Web App**
To start the web server:
```bash
python app.py
```
Open your web browser and navigate to **`http://127.0.0.1:5000`** to use the application.

---

## 📸 Screenshots

### CNN Model Performance
![CNN Training Output](<https://imgur.com/9JJ3YIt.png>)
*Terminal output showing the final training epoch and a test accuracy above 98%.*

### Flask Web Application
![Flask App Screenshot](<https://imgur.com/pND8iNT.png>)
*The web interface with a hand-drawn digit and the model's correct prediction.*

### NLP Analysis Output
![NLP Output Screenshot](<https://imgur.com/ZaWDhWW.png>)
*Terminal output showing extracted product names and positive/negative sentiment analysis.*

### CONCLUSION

This project successfully demonstrates a full-stack approach to AI development, bridging the gap between theoretical models and practical, interactive applications.

### Project Contributors
*MAKAMU OKINYI BETSY* - makamubetsy@gmail.com

*BRIDIE MAUGHAM DIBORA* - maughamdiborapr@gmail.com
