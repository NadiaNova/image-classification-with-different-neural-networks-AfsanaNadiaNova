# Step 1: Importing the libraries and loading the dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,  GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.optimizers import Adam

# Load the dataset
train_datasets = "/kaggle/input/brain-tumor-classification-mri/Training"
validation_datasets = "/kaggle/input/brain-tumor-classification-mri/Training"

batch_size = 64
image_size = 224
epochs = 5

# Step 2: Create data generators for training and Validations
def prepare_the_datasets(train_datasets, validation_datasets, batch_size, image_size):

    train_datasets_generator = ImageDataGenerator(rescale=1./255,
                                                  shear_range = 0.2, 
                                                  zoom_range = 0.2, 
                                                  horizontal_flip = True, 
                                                  fill_mode = "nearest")


    validation_datasets_generator = ImageDataGenerator(rescale=1.0/255)


    train_datasets_generator_data = train_datasets_generator.flow_from_directory(
        batch_size = batch_size,
        directory = train_datasets,
        shuffle = True,
        target_size = (image_size, image_size),
        class_mode = "categorical"
    )

    validation_datasets_generator_data = validation_datasets_generator.flow_from_directory(
        batch_size = batch_size,
        directory = validation_datasets,
        shuffle = True,
        target_size = (image_size, image_size),
        class_mode = "categorical"
    )


    return train_datasets_generator_data, validation_datasets_generator_data


# Step 3: prepare the datasets
train_data , validation_data = prepare_the_datasets(train_datasets, validation_datasets, batch_size, image_size)

# Step 4: Create the Sequencial model
model = Sequential([
    Conv2D(32, (3, 3), activation = 'relu', input_shape = (image_size, image_size, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation = 'relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation = 'relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    
    Dense(512, activation = "relu"),
    Dense(4, activation = "softmax")
])


#using optimizer
model.compile(optimizer="adam", 
              loss = "categorical_crossentropy",
              metrics = ["accuracy"]
              )


model_checkpoint_filpath = "model_checkpoint.h5"
callbacks_checkpoints = ModelCheckpoint(

    filepath = model_checkpoint_filpath, 
    save_weights_only = True, 
    monitor = "val_accuracy",
    mode = "max", 
    save_best_only = True
)


history = model.fit(train_data, 
                    steps_per_epoch = len(train_data),
                    epochs = epochs,
                    validation_data = validation_data, 
                    validation_steps = len(validation_data),
                    callbacks = [callbacks_checkpoints]
                    )


#finding the loss and accuracy
loss, accuracy = model.evaluate(validation_data, batch_size=batch_size)


# Step 5: Create the DenseNet201 model

# Create the DenseNet201 model
base_model = DenseNet201(weights='imagenet', include_top=False)
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(4, activation='softmax'))


# Freeze the layers of the pre-trained DenseNet201 model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# Train the model
history = model.fit(train_data, 
                    steps_per_epoch = len(train_data),
                    epochs = epochs,
                    validation_data = validation_data, 
                    validation_steps = len(validation_data),
                    callbacks = [callbacks_checkpoints]
                    )


#Find the Loss and Accuracy for DenseNet201 Model
loss, accuracy = model.evaluate(validation_data, batch_size=batch_size)


# Plot training history: plotting accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Plot the loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
