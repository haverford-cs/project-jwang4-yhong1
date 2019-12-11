import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import util


def create_model():
    model = tf.keras.models.Sequential([
                # Adds a densely-connected layer with 64 units to the model:
                tf.keras.layers.Dense(64, activation='relu'),
                # Add another:
                tf.keras.layers.Dense(64, activation='relu'),
                # Add a softmax layer with 10 output units:
                tf.keras.layers.Dense(2, activation='softmax')])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    opts = util.parse_args()
    X, y = util.data_load(opts.dataset)
    model = create_model()
    ratio = opts.upsamplet
    n = opts.upsamplen
    if ratio is not None and n is not None:
        print("User can only choose one type of upsample")
        sys.exit()
    elif ratio is not None:
        needed = util.needed_total(X, y, ratio)
        X, y = util.upsample(X, y, needed)
    elif n is not None:
        needed = util.needed_n(X, y, n)
        X, y = util.upsample(X, y, needed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_test = util.normalize(X_train, X_test)
    train_dset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64, drop_remainder=False).shuffle(buffer_size=10000)
    test_dset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)
    model.fit(train_dset, epochs=10)
    conf_mat = np.zeros((2, 2), dtype=int)
    for d, labels in test_dset:
        predictions = model(d)
        for i in range(len(d)):
            conf_mat[labels[i]][np.argmax(predictions[i])] += 1
    print(conf_mat)


if __name__ == '__main__':
    main()
