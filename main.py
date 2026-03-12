import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Setup Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 10

# 2. Data Augmentation (Crucial for "Advanced" detection)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'dataset_folder/', # Replace with your dataset path
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'dataset_folder/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 3. Transfer Learning: Load Pre-trained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False, # Remove the final ImageNet classification layer
    weights='imagenet'
)
base_model.trainable = False  # Freeze the base layers

# 4. Build the Custom Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train
model.fit(train_generator, validation_data=validation_generator, epochs=EPOCHS)

# 6. Save for deployment
model.save('plant_disease_model.h5')
print("Model Saved!")