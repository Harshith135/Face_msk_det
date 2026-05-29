import numpy as np, os, cv2, kagglehub, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

path = kagglehub.dataset_download('omkargurav/face-mask-dataset')
data_path = os.path.join(path, 'data')
classes = ['without_mask', 'with_mask']
img_size = 128
X, y = [], []
for cls in classes:
    folder = os.path.join(data_path, cls)
    label = classes.index(cls)
    for f in os.listdir(folder):
        try:
            img = cv2.cvtColor(cv2.imread(os.path.join(folder, f)), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(label)
        except:
            pass
X = np.array(X)/255.0
y = np.array(y)

X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.5,random_state=42,stratify=y_temp)

train_gen = ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.15,horizontal_flip=True,fill_mode='nearest').flow(X_train,y_train,batch_size=32,shuffle=True)
val_gen = ImageDataGenerator().flow(X_val,y_val,batch_size=32,shuffle=False)
test_gen = ImageDataGenerator().flow(X_test,y_test,batch_size=32,shuffle=False)

model = Sequential([
Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
BatchNormalization(), MaxPooling2D(2,2),
Conv2D(64,(3,3),activation='relu'), BatchNormalization(), MaxPooling2D(2,2),
Conv2D(128,(3,3),activation='relu'), BatchNormalization(), MaxPooling2D(2,2),
Flatten(), Dense(128,activation='relu'), Dropout(0.5), Dense(1,activation='sigmoid')])

model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
callbacks=[EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=2)]
history = model.fit(train_gen, validation_data=val_gen, epochs=15, callbacks=callbacks)

loss, acc = model.evaluate(test_gen)
print('Test Accuracy:', acc)
y_pred = (model.predict(test_gen) > 0.5).astype(int).reshape(-1)
print(classification_report(y_test, y_pred, target_names=['Without Mask','With Mask']))
model.save('face_mask_cnn_model.keras')
print('Saved model successfully!')