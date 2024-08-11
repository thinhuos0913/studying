# Plot Loss and Accuracy line:
# Đầu tiên ta lưu lịch sử train vào biến history
# history = model.fit_generator(aug.flow(X_train, y_train, batch_size=64), epochs=50, validation_data=aug.flow(X_test,y_test, batch_size=128))

# Tiếp theo ta plot các thông số loss và acc ra
# def plot_model_history(model_history, acc='accuracy', val_acc='val_accuracy'):
#     fig, axs = plt.subplots(1, 2, figsize=(15, 5))
#     axs[0].plot(range(1, len(model_history.history[acc]) + 1), model_history.history[acc])
#     axs[0].plot(range(1, len(model_history.history[val_acc]) + 1), model_history.history[val_acc])
#     axs[0].set_title('Model Accuracy')
#     axs[0].set_ylabel('Accuracy')
#     axs[0].set_xlabel('Epoch')
#     axs[0].set_xticks(np.arange(1, len(model_history.history[acc]) + 1), len(model_history.history[acc]) / 10)
#     axs[0].legend(['train', 'val'], loc='best')
#     axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
#     axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
#     axs[1].set_title('Model Loss')
#     axs[1].set_ylabel('Loss')
#     axs[1].set_xlabel('Epoch')
#     axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
#     axs[1].legend(['train', 'val'], loc='best')
#     plt.show()
#     plt.savefig('roc.png')

# plot_model_history(history)

# EXAMPLE: IRIS Classification
# import thư viện
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Thực hiện load dữ liệu
iris_data = load_iris() 

# In ra 10 input đầu tiên
print('First 10 inputs: ')
print(iris_data.data[:10])
# In ra 10 output đầu tiên
print('First 10 output (label): ')
print(iris_data.target[:10])

# Gán input vào biến X
X = iris_data.data
# Gán output vào biến y 
y = iris_data.target.reshape(-1,1)

# print(y)
# Thực hiện Onehot transform
# encoder = OneHotEncoder(sparse=False)
# y = encoder.fit_transform(y)
# print("Output after transform")
# # print(y)

# Chia dữ liệu train, test với tỷ lệ 80% cho train và 20% cho test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(y_test)
y_test = y_test.reshape(30,)
print(y_test)
# Kiểm tra trên tập test
results = model.score(X_test, y_test)
print(results)
# print('Test loss: {:4f}'.format(results[0]))
# print('Test accuracy: {:4f}'.format(results[1]))

# Evaluate predictions
#print(predictions)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# Use Neural Nets
# import numpy as np

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam

# # import thư viện
# from sklearn.datasets import load_iris
# # Thực hiện load dữ liệu
# iris_data = load_iris() 

# # In ra 10 input đầu tiên
# print('First 10 inputs: ')
# print(iris_data.data[:10])
# # In ra 10 output đầu tiên
# print('First 10 output (label): ')
# print(iris_data.target[:10])

# # Gán input vào biến X
# X = iris_data.data
# # Gán output vào biến y 
# y = iris_data.target.reshape(-1,1)

# # Thực hiện Onehot transform
# encoder = OneHotEncoder(sparse=False)
# y = encoder.fit_transform(y)
# print("Output after transform")
# print(y[:10])

# # Chia dữ liệu train, test với tỷ lệ 80% cho train và 20% cho test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# # Khai báo model
# model = Sequential()

# model.add(Dense(128, input_shape=(4,), activation='relu', name='layer1'))
# model.add(Dense(128, activation='relu', name='layer2'))
# model.add(Dense(3, activation='softmax', name='output'))

# # Cài đặt hàm tối ưu Adam 
# optimizer = Adam()
# model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# # In cấu trúc mạng ra màn hình
# print('Detail of network: ')
# print(model.summary())

# # Train model
# model.fit(X_train, y_train, batch_size=32, epochs=10)

# # Kiểm tra trên tập test
# results = model.evaluate(X_test, y_test)

# print('Test loss: {:4f}'.format(results[0]))
# print('Test accuracy: {:4f}'.format(results[1]))

# # Train model
# import matplotlib.pyplot as pyplot
# history = model.fit(X_train, y_train, batch_size=32, epochs=200,validation_data=(X_test,y_test))

# # plot loss during training

# pyplot.figure(figsize=(20,10))
# pyplot.subplot(211)
# pyplot.title('Loss')
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# # plot accuracy during training
# pyplot.subplot(212)
# pyplot.title('Accuracy')
# pyplot.plot(history.history['accuracy'], label='train')
# pyplot.plot(history.history['val_accuracy'], label='test')
# pyplot.legend()
# pyplot.show()

# # demonstration of calculating metrics for a neural network model using sklearn
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import confusion_matrix

# y_hat = model.predict(X_test)
# y_pred = np.argmax(y_hat, axis=1)
# y_test_label =  np.argmax(y_test, axis=1)


# # accuracy: (tp + tn) / (p + n)
# accuracy = accuracy_score(y_test_label, y_pred)
# print('Accuracy: %f' % accuracy)
# # precision tp / (tp + fp)
# precision = precision_score(y_test_label, y_pred, average='macro')
# print('Precision: %f' % precision)
# # recall: tp / (tp + fn)
# recall = recall_score(y_test_label, y_pred, average='macro')
# print('Recall: %f' % recall)
# # f1: 2 tp / (2 tp + fp + fn)
# f1 = f1_score(y_test_label, y_pred, average='macro')
# print('F1 score: %f' % f1)

# auc = roc_auc_score(y_test, y_hat, multi_class='ovr')
# print('ROC AUC: %f' % auc)
# # confusion matrix
# matrix = confusion_matrix(y_test_label, y_pred)
# print(matrix)

# import pandas as pd
# import seaborn as sn
# df_cm = pd.DataFrame(matrix, index = [i for i in "012"],
#                   columns = [i for i in "012"])
# pyplot.figure(figsize = (10,7))
# sn.heatmap(df_cm, annot=True)