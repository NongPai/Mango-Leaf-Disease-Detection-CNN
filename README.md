# Mango Leaf Disease Detection & Classification 🥭

A deep learning-based system designed to identify and classify mango leaf diseases using **CNN (EfficientNetB0)** and **CLIP** for pre-verification, featuring a real-time web interface built with **Gradio**.

## 🌟 Key Features
- **High-Accuracy Classification:** Detects 7 classes of mango leaf conditions (including healthy) with an overall accuracy of **99.8%**.
- **Pre-verification with CLIP:** Utilizes OpenAI's CLIP model to verify if the uploaded image is a "mango leaf" before processing, reducing false positives.
- **Real-Time Diagnosis:** Provides instant feedback via a user-friendly web interface.
- **Agricultural Recommendations:** Offers actionable guidance on disease management (e.g., pruning, fungicide application) based on the diagnosis results.

## 📁 Project Structure
The repository is organized into two main modules:
- `/training`: Contains source code for model development, including data augmentation, fine-tuning of the EfficientNetB0 architecture, and performance evaluation.
- `/webapp`: Contains the deployment code, including the Gradio web interface, CLIP pre-verification logic, and the integration of the knowledge base for recommendations.

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.x installed. It is recommended to use a virtual environment.

### 2. Installation
Install the required libraries using the following command:
```bash
pip install -r requirements.tx
```
### 3. Running the Web Application
Navigate to the webapp directory and execute the main script:
```bash
cd webapp
python app.py
```

🛠️ Technology Stack
- Deep Learning Framework: TensorFlow / Keras (EfficientNetB0)
- Zero-Shot Classification: OpenAI CLIP (for input verification)
- Web Interface: Gradio Framework
- Data Processing: OpenCV, NumPy, Pandas
- Development Language: Python
  
📄 License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

👥 Contributors
NongPai (Project Lead & Developer)
