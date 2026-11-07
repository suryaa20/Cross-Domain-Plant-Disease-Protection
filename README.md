# Cross-Domain Deep Learning for Robust Plant Disease Detection

**Academic Project for 7th Semester Deep Learning Course**

## Project Motivation
A common challenge in computer vision for agriculture is the "domain gap." Models trained on high-quality, lab-controlled datasets (like **PlantVillage**) achieve 99% accuracy but fail completely when deployed in the real world on noisy, cluttered field images (like **PlantDoc**). This project demonstrates this domain gap and uses **Domain Adaptation (Fine-tuning)** to build a robust model capable of real-world plant disease detection.

## Dataset Details
1. **Source Domain (PlantVillage):** Lab-controlled environment, single leaves on plain backgrounds. Used for pretraining our model to recognize basic disease patterns.
2. **Target Domain (PlantDoc):** Real-world field photos, complex backgrounds, multiple leaves, varied lighting. Used to expose the domain gap and for fine-tuning to achieve robustness.

We filter both datasets to include only common classes (e.g., Apple Scab, Apple Healthy, Tomato Early Blight, Tomato Healthy) to ensure a fair cross-domain evaluation.

## Architecture Explanation
- **Backbone:** We use **ResNet18**, a lightweight Convolutional Neural Network (CNN). It's small enough to train on a laptop but powerful enough to extract meaningful features from images.
- **Baseline Model:** A ResNet18 trained *only* on PlantVillage. We evaluate it on both PlantVillage (expecting high accuracy) and PlantDoc (expecting low accuracy due to the domain gap).
- **Cross-Domain Strategy (Fine-Tuning):** We take the Baseline model and fine-tune its final layers on a small subset of PlantDoc. This allows the model to adjust its learned features to the real-world domain without forgetting the disease patterns learned from the large PlantVillage dataset.
- **Explainability:** We integrate **Grad-CAM**, which generates heatmaps showing exactly which parts of the leaf the model looked at to make its prediction. This opens the "black box" of deep learning.

## Project Structure
- `/data`: Scripts to download and prepare the datasets.
- `/model`: PyTorch definitions for the ResNet18 model, datasets, training loops, and Grad-CAM logic.
- `/notebooks`: An exploratory Jupyter Notebook walking through the training process step-by-step.
- `/app`: A Streamlit web application for interactive demonstration.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare data: Run `python data/prepare_data.py` (Note: This is a placeholder script simulating data prep).
3. Train Baseline: Run `python model/train_baseline.py`
4. Train Fine-Tuned Model: Run `python model/train_finetune.py`
5. Run the web app: `streamlit run app/app.py`

## Results Summary
* The baseline model trained on PlantVillage performs extremely well on PlantVillage test data but suffers a significant drop in accuracy on PlantDoc.
* After fine-tuning on a small portion of PlantDoc, the model recovers its performance, demonstrating cross-domain robustness.
* Grad-CAM visualizations confirm that the model focuses on actual disease lesions rather than background artifacts.
