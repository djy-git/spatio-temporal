from common import *

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.callbacks import EarlyStopping

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == '__main__':
    # 1. Load dataset
    full_data = pd.read_csv(join(PATH.input, 'train_data.csv'))
    sample_submission = pd.read_csv(join(PATH.input, 'sample_submission.csv'))
    full_data = full_data.fillna(method = 'bfill')


    # 2. Preprocessing
    tms_list = list(pd.unique(full_data['Tmstamp']))
    full_data['Tmstamp'] = full_data['Tmstamp'].apply(lambda x: tms_list.index(x)+1)

    ## 2.1 Split data
    SEQ_LEN = 5
    train_x, train_y = make_train_data(full_data, SEQ_LEN)
    val_x,   val_y   = make_val_data(full_data, SEQ_LEN)
    test_x,  _       = make_test_data(full_data, SEQ_LEN)
    val_x,   val_y   = val_x.reshape(-1, *train_x.shape[1:]), val_y.reshape(-1, train_y.shape[1])
    test_x           = test_x.reshape(-1, *train_x.shape[1:])
    print("Train data(X, y):", train_x.shape, train_y.shape)
    print("Val data(X, y):", val_x.shape, val_y.shape)
    print("Test data(X):", test_x.shape)

    ## 2.2 Generate dataset
    BATCH_SIZE = 128
    train_ds = generate_dataset(train_x, train_y, batch_size=BATCH_SIZE)
    val_ds   = generate_dataset(val_x, val_y, batch_size=BATCH_SIZE)
    test_ds  = generate_dataset(test_x, batch_size=BATCH_SIZE)


    # 3. Modeling
    model = Sequential([
        GRU(256, input_shape=train_x[0].shape),
        Dense(516, activation='relu'),
        Dense(288, activation='relu')
    ])
    optimizer = tf.optimizers.RMSprop(0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])


    # 4. Training
    model.fit(train_ds, validation_data=val_ds, epochs=1000, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])


    # 5. Predict
    sample_submission['Patv'] = model.predict(test_ds).reshape(-1)
    submission_path = join(PATH.output, f"{basename(__file__).split('.')[0]}.csv")
    sample_submission.to_csv(submission_path, index=False)
    print("Submission file is saved to", submission_path)
