import keras


class PoseCNNModel(keras.Model):
    def __init__(self, units=30, activation="relu"):
        pass

    '''
n_classes = len(np.unique(pipeline.y_encoder))
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model = keras.Sequential([
    keras.layers.Input(shape=(50, 99)),
    keras.layers.Conv1D(64, 3, activation="relu"),
    keras.layers.GlobalMaxPooling1D(),
    keras.layers.Dense(n_classes, activation="softmax")
])


optimizer = keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01)
model.compile(loss=["sparse_categorical_crossentropy"], optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=[callback])
'''
