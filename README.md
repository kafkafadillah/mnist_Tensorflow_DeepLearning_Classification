# mnist_Tensorflow_DeepLearning_Classification
Deep Learning Classification using mnist Tensorflow dataset

```markdown
# 🧠 MNIST Digit Classification with TensorFlow & Keras

This project is a simple deep learning implementation for classifying handwritten digits (0-9) using the **MNIST** dataset. The dataset consists of 60,000 training images and 10,000 testing images, each of size 28x28 pixels.

## 🚀 Technologies Used

- Python 3.x
- TensorFlow & Keras
- NumPy
- Matplotlib
- Google Colab / Jupyter Notebook

## 📁 Dataset Structure

The MNIST dataset is loaded directly from `tensorflow.keras.datasets` and includes:

- `gambar_latih` (X_train): 28x28 grayscale images (60,000 training samples)
- `label_latih` (y_train): Integer labels from 0 to 9
- `gambar_testing` (X_test): Testing images (10,000 samples)
- `label_testing` (y_test): Testing labels

## 🧩 Model Architecture

The model uses a basic feedforward neural network:

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

- **Optimizer**: Adam  
- **Loss Function**: Sparse Categorical Crossentropy  
- **Evaluation Metric**: Accuracy

## ✅ Custom Callback

A custom callback is used to stop training early once accuracy exceeds 90%:

```python
class MyCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.9:
            print('\nAccuracy reached 90%, stopping training!')
            self.model.stop_training = True
```

## 📊 Results & Visualization

After training, the loss and accuracy are visualized using Matplotlib:

```python
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train'], loc='best')
plt.show()

plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train'], loc='best')
plt.show()
```

## 📌 How to Run

1. Clone this repository or open the notebook in Google Colab.
2. Run each cell sequentially.
3. The model will train for a few epochs and stop automatically when it reaches 90% accuracy.
4. You can evaluate the model on test data using `model.evaluate()`.

---

Feel free to improve and build on this model — try adding more layers, regularization, or even CNNs for better performance!
