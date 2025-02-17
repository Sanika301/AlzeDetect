import os
import zipfile
import cv2
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Parameters for LBP
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

# Function to extract dataset from zip file
def extract_dataset(zip_path, extract_to="dataset"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return os.path.join(extract_to, "2DSagittal")

# Function to preprocess images and apply augmentation
def preprocess_images(image_dir):
    images = []
    labels = []

    label_mapping = {"AD": 0, "CN": 1, "MCI": 2}  # Mapping folder names to labels

    for label_dir in os.listdir(image_dir):
        label_path = os.path.join(image_dir, label_dir)
        if not os.path.isdir(label_path) or label_dir not in label_mapping:
            continue

        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Skull stripping using adaptive thresholding
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            filtered = cv2.medianBlur(binary, 5)
            image = cv2.bitwise_and(image, image, mask=filtered)

            # Resize to standard dimensions
            image = cv2.resize(image, (128, 128))

            images.append(image)
            labels.append(label_mapping[label_dir])

    images = np.array(images)
    labels = np.array(labels)

    # Normalize images (zero mean, unit variance)
    mean = np.mean(images, axis=(0, 1, 2), keepdims=True)
    std = np.std(images, axis=(0, 1, 2), keepdims=True)
    images = (images - mean) / std

    return images, labels

# Function to apply data augmentation
def augment_images(images, labels):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    augmented_images = []
    augmented_labels = []

    for i in range(len(images)):
        image = np.expand_dims(images[i], axis=-1)  # Ensure the shape is (128, 128, 1)
        label = labels[i]

        # Generate augmented images
        image = np.expand_dims(image, axis=0)  # Add batch dimension (shape becomes (1, 128, 128, 1))
        for batch in datagen.flow(image, batch_size=1):
            augmented_images.append(batch[0].astype(np.float32))
            augmented_labels.append(label)
            break  # Stop after one augmentation to avoid infinite loop

    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    return augmented_images, augmented_labels

# Function to extract LBP features
def extract_lbp_features(images):
    lbp_features = []
    for image in images:
        image = (image * 255).astype(np.uint8)  # Convert to integer type
        lbp = local_binary_pattern(image, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
        lbp_hist = lbp_hist.astype(np.float32)
        lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize histogram
        lbp_features.append(lbp_hist)
    return np.array(lbp_features)

# Main script
if __name__ == "__main__":
    zip_path = "/content/2DSagittal.zip"
    dataset_dir = extract_dataset(zip_path)

    # Preprocess images
    images, labels = preprocess_images(dataset_dir)

    # Perform data augmentation
    augmented_images, augmented_labels = augment_images(images, labels)

    # Feature extraction
    lbp_features = extract_lbp_features(augmented_images.squeeze())

    # Dimensionality reduction
    max_components = min(lbp_features.shape[0], lbp_features.shape[1])  # Ensure valid n_components
    n_components = min(50, max_components)  # Use 50 or the maximum feasible value
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(lbp_features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, augmented_labels, test_size=0.2, stratify=augmented_labels, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Base models for stacking
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
    ada_model = AdaBoostClassifier(n_estimators=200, random_state=42)
    svm_model = SVC(probability=True, kernel='rbf', C=10, gamma=0.1, random_state=42)

    # Train base classifiers and evaluate
    models = [rf_model, gb_model, ada_model, svm_model]
    model_names = ["Random Forest", "Gradient Boosting", "AdaBoost", "SVM"]

    for model, name in zip(models, model_names):
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # Calculate accuracy
        train_accuracy = accuracy_score(y_train, train_preds)
        test_accuracy = accuracy_score(y_test, test_preds)

        print(f"{name} - Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")

    # Predictions from base classifiers
    rf_preds = rf_model.predict_proba(X_test)
    gb_preds = gb_model.predict_proba(X_test)
    ada_preds = ada_model.predict_proba(X_test)
    svm_preds = svm_model.predict_proba(X_test)

    # Combine predictions
    stacked_features = np.hstack([rf_preds, gb_preds, ada_preds, svm_preds])

    # Meta-learner
    meta_learner = GradientBoostingClassifier(n_estimators=200, random_state=42)
    meta_learner.fit(np.hstack([rf_model.predict_proba(X_train),
                                gb_model.predict_proba(X_train),
                                ada_model.predict_proba(X_train),
                                svm_model.predict_proba(X_train)]), y_train)

    # Evaluate Stacking Model
    final_train_preds = meta_learner.predict(np.hstack([rf_model.predict_proba(X_train),
                                                        gb_model.predict_proba(X_train),
                                                        ada_model.predict_proba(X_train),
                                                        svm_model.predict_proba(X_train)]))
    final_test_preds = meta_learner.predict(stacked_features)

    stacking_train_acc = accuracy_score(y_train, final_train_preds)
    stacking_test_acc = accuracy_score(y_test, final_test_preds)

    print(f"Stacking Model - Train Accuracy: {stacking_train_acc:.2f}, Test Accuracy: {stacking_test_acc:.2f}")

    joblib.dump(rf_model, 'rf_model.pkl')
    joblib.dump(gb_model, 'gb_model.pkl')
    joblib.dump(ada_model, 'ada_model.pkl')
    joblib.dump(svm_model, 'svm_model.pkl')

    # Save the meta-learner (stacking model)
    joblib.dump(meta_learner, 'meta_learner.pkl')

    print("Models saved successfully!")
        # Display predicted probabilities and labels for a sample of test data
    sample_indices = np.random.choice(len(X_test), 10, replace=False)  # Randomly select 10 samples
    sample_predictions = meta_learner.predict(stacked_features[sample_indices])
    sample_probabilities = meta_learner.predict_proba(stacked_features[sample_indices])

    print("\nSample Predictions and Probabilities:")
    for i, index in enumerate(sample_indices):
        print(f"Sample {i+1}:")
        print(f"True Label: {y_test[index]}")
        print(f"Predicted Label: {sample_predictions[i]}")
        print(f"Probabilities: {sample_probabilities[i]}\n")

    # Confusion matrix
    cm = confusion_matrix(y_test, final_test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["AD", "CN", "MCI"], yticklabels=["AD", "CN", "MCI"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, final_test_preds, target_names=["AD", "CN", "MCI"]))

