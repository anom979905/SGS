import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


print("Current Working Directory:", os.getcwd())

# Set environment variable for TensorFlow optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

def gen_labels():
    train_dir = '../Data/Train'
    
    # Check if the directory exists
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"The directory '{train_dir}' does not exist. Please check the path.")
    
    # Initialize the ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1/255)
    
    # Create a generator to read images from the directory
    try:
        train_generator = datagen.flow_from_directory(
            train_dir,
            target_size=(300, 300),
            batch_size=32,
            class_mode='sparse'
        )
    except Exception as e:
        raise RuntimeError(f"An error occurred while creating the image data generator: {e}")

    # Extract class labels
    labels = train_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())
    
    return labels

def preprocess(image):
    image = np.array(image.resize((300, 300), Image.Resampling.LANCZOS))
    image = np.array(image, dtype='uint8')
    image = np.array(image)/255.0
    return image

def model_arc():
    model = Sequential()

    # Convolution blocks
    model.add(Conv2D(32, kernel_size=(3,3), padding='same', input_shape=(300,300,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    # Classification layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

