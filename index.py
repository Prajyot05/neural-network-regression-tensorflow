import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

# Features
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Labels
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# 1. Create the model using sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

# 2. Compile the model (mae -> mean absolute error, SGD -> stochastic gradient descent)
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# 3. Fit the model (5 chances)
model.fit(tf.expand_dims(X, axis=-1), y, epochs=5)

# 4. Test the accuracy of the model's output
y_pred = model.predict(np.array([16.0]))
print(y_pred)

# The accuracy of the model at this point is not good at all