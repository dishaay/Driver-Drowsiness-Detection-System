import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "archive/mrleyedataset"

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 5


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

print("Classes:", train_dataset.class_names)


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# CNN
# ============================================================

model = models.Sequential([

    # Convert pixel values 0-255 → 0-1
    layers.Rescaling(
        1.0 / 255,
        input_shape=(IMG_SIZE, IMG_SIZE, 1)
    ),

    # First feature extraction block
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Second feature extraction block
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Third feature extraction block
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convert feature maps → vector
    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    # Helps reduce overfitting
    layers.Dropout(0.3),

    # Binary classification
    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\nTraining CNN...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    "drowsiness_cnn.keras"
)

print("\nMODEL SAVED!")
print("File: drowsiness_cnn.keras")