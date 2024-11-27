import os
import requests
from tqdm import tqdm

def download_model(url, save_path):
    """
    Download a model file from URL and save to specified path
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

# Model URLs
MOBILENET_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v2_1.0_224_quant_and_labels.zip"
FACE_DETECTION_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

# Download paths
PERSON_MODEL_PATH = 'model/mobilenet_v2_thermal.tflite'
FACE_MODEL_PATH = 'model/face_detection_model.tflite'

def setup_models():
    """
    Download and setup both required models
    """
    print("Downloading MobileNet v2 model...")
    download_model(MOBILENET_URL, PERSON_MODEL_PATH)
    
    print("Downloading Face Detection model...")
    download_model(FACE_DETECTION_URL, FACE_MODEL_PATH)
    
    print("Models downloaded successfully!")
    
if __name__ == "__main__":
    setup_models()