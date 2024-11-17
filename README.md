# Meesho-Attribute-Predict-Competition

## Collaborators

- [Sunny Kumar](https://github.com/Epoch-Seeker)
- [sowhatnowgithub](https://github.com/sowhatnowgithub)
- [Shresth](https://github.com/notsocoolshresth)
- [Vedalaxman](https://github.com/Vedalaxman)

> Please change the path directories accordingly in each file.

## Setup Instructions

### Prerequisites
```bash
python==3.7
tensorflow==2.6.1
keras==3.3.3
opencv-python==4.10.0
scikit-learn==1.2.2
numpy==1.21.6
pandas==2.2.3
matplotlib==3.7.5
```

### Environment Setup

1. Clone the repository:
```bash

cd clothing-classification
```

2. Set up Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Organization and Preprocessing

### Directory Structure
```
project/
├── data.md
├── men_tshirts.ipynb
├── sarees.ipynb
├── kurtis.ipynb
├── women_tshirts.ipynb
└── women_tops_and_tunics.ipynb
└── README.md
```

### Data Preprocessing Steps


Our model first takes downloads the datasets from the kaggle api and then convert them into separate categories and split the train.csv to corresponding categories of csv
Please follow the data.md for further clarifications.


## Model Architecture of starting.py

Execute the starting.py and get the csv files of all the individuals

1. Load and preprocess the data:
```python
# Example preprocessing script (src/preprocess.py)
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def preprocess_data():
    # Read main training data
    df = pd.read_csv('data/raw/train.csv')
    
    # Define categories
    categories = [
        'Men Tshirts', 
        'Sarees', 
        'Kurtis', 
        'Women Tshirts', 
        'Women Tops & Tunics'
    ]
    
    # Split data by category
    for category in categories:
        filtered_df = df[df['Category'] == category]
        filename = f"data/processed/{category.lower().replace(' ', '_').replace('&', 'and')}.csv"
        filtered_df.to_csv(filename, index=False)
        print(f"Saved {filename} with {len(filtered_df)} rows.")
```

2. Image preprocessing:
```python
def preprocess_image(image_path):
    # Read and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img
```





### Model Architecture
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Dropout(0.2),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(num_classes, activation='softmax')
])
```

### Training Parameters
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy, F1-Score
- Batch Size: 32
- Epochs: 50

## Inference

1. Generate predictions:

First downloads the weights of the model

These are the weights of the model.

### Model Weights

Download trained weights from:
[https://drive.google.com/file/d/14aHzv6mW0jXbE6NimngNdjXqeSdIhxgF/view?usp=sharing]


Once weights are downloaded, open a categorie of one the notebook and then loads the weight and execute
## Reproducibility


### Hardware Requirements
- Recommended: GPU with 6GB+ VRAM
- Minimum: 16GB RAM for CPU training
- Storage: 20GB free space


## Troubleshooting

1. Memory Issues:
   ```python
   # Reduce memory usage
   batch_size = 16  # Reduce batch size
   tf.keras.backend.clear_session()  # Clear GPU memory
   ```

2. GPU Issues:
   ```bash
   # Set GPU memory growth
   export TF_GPU_ALLOCATOR=cuda_malloc_async
   ```

