from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# dataset path
DATASET_PATH = "dataset"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30

# reading and splitting images, applying augmentation to training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    brightness_range=(0.5, 1.5),
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)

class_weights = dict(enumerate(class_weights))


# creating the model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

#early stopping adding (new)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss", mode="min")
#training the model
#history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[early_stop])

# Fine Tuning
base_model.trainable = True  # Unfreezing the base model

# Fine-tuning the top layers
for layer in base_model.layers[:-40]:  # Unfreezing the last 20 layers
    layer.trainable = False

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
#history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[early_stop, reduce_lr, checkpoint], class_weight=class_weights)

# Accuracy in the latest epoch
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")  
print(f"Final Validation Accuracy: {final_val_acc:.4f}")


# loading test set
test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# calculating accuracy of the test set
loss, acc = model.evaluate(test_gen)
print(f"🧪 Test Accuracy: {acc:.4f}")

# class names
class_names = list(test_gen.class_indices.keys())

# model predictions
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# F1-score (weighted average)
f1 = f1_score(y_true, y_pred_classes, average='weighted')
print(f"\n✅ Weighted F1 Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

#save model
model.save('waste_classifier_model.h5')
print("Model file saved: waste_classifier_model.h5")