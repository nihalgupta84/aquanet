import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from PIL import Image

def extract_hsv_features(image_paths, size=(64, 64)):
    """Extract HSV color histograms and stats for classical ML models."""
    features = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        img = cv2.resize(img, size)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Histograms
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
        
        # Mean/Std stats
        mean_stats = hsv.mean(axis=(0, 1))
        std_stats = hsv.std(axis=(0, 1))
        
        feat = np.concatenate([h_hist, s_hist, v_hist, mean_stats, std_stats])
        features.append(feat)
        
    return np.array(features)

def get_classical_model(model_name):
    """Instantiate classical machine learning baselines."""
    if model_name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'knn':
        return KNeighborsClassifier(n_neighbors=5)
    elif model_name == 'svm':
        return SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    else:
        raise ValueError(f"Unknown classical model: {model_name}")
