import numpy as np
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation


ind = np.load('saved/indoor_X_train_.npy')
out = np.load('saved/outdoor_X_train_.npy')
tran = np.load('saved/transportation_X_train_.npy')

X_test = np.load('saved/_X_test_.npy')
y_test = np.load('saved/_y_test_.npy')

X_train2 = np.concatenate([ind, out], axis = 0)
y_train2 = np.concatenate([[0]*len(ind), [1]*len(out)], axis = 0)
print(X_train2.shape, y_train2.shape)
X_test2, y_test2 = [], []
for i, label in enumerate(y_test):
	if label != 2:
		y_test2.append(label)
		X_test2.append(X_test[i])

X_train2 = np.log(X_train2)/np.log(20)
X_test2 = np.log(X_test2)/np.log(20)
X_train2 = [(i - np.min(i))/(np.max(i) - np.min(i)) for i in X_train2]
X_test2 = [(i - np.min(i))/(np.max(i) - np.min(i)) for i in X_test2]

X_train2 = np.array(X_train2)
y_train2 = np.array(y_train2)
X_test2 = np.array(X_test2)
y_test2 = np.array(y_test2)

print(X_test2.shape, y_test2.shape)
y_train2_onehot = to_categorical(y_train2)
y_test2_onehot = to_categorical(y_test2)
X_train2 = np.array([i[..., np.newaxis] for i in X_train2])
X_test2 = np.array([i[..., np.newaxis] for i in X_test2])

# add noise
X_tmp = np.copy(X_train2)
X_train2 = []
for i in X_tmp:
	if np.random.random() > 0.5:
		X_train2.append(i + np.random.normal(0,1,i.shape) * 1e-3)
	else:
		X_train2.append(i)
X_train2 = np.array(X_train2)

X_test2 = np.array([i + np.random.normal(0,1,i.shape) * 1e-3 for i in X_test2])

#create model
model = Sequential()
#add model layers
model.add(Conv2D(32, kernel_size = (3,1), activation = 'relu', input_shape=(60,8,1)))
model.add(BatchNormalization())
# model.add(MaxPooling2D(3,1))
model.add(Dropout(0.3))
model.add(Conv2D(16, kernel_size = (3,1), activation = 'relu'))
model.add(BatchNormalization())
# model.add(MaxPooling2D(3,1))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=0, mode='max')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='max')

model.fit(X_train2, y_train2_onehot, validation_data=(X_test2, y_test2_onehot), epochs = 50, verbose = True, shuffle = True, batch_size = 32, callbacks=[earlyStopping])

y_pred = model.predict(X_test2)
y_pred_label = np.argmax(y_pred, axis = 1)

print(classification_report(y_test2, y_pred_label))
print(confusion_matrix(y_test2, y_pred_label))

model.save('ind_vs_out.keras.model')
