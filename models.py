# DENSE MODEL
dense_model = Sequential([
    Dense(128, input_shape=(586,), activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# LSTM MODEL
lstm_model = Sequential([
    LSTM(64, input_shape=(586, 1), activation='tanh', recurrent_activation='sigmoid', return_sequences=True),
    BatchNormalization(),
    Dropout(0.5),
    LSTM(64, activation='tanh', recurrent_activation='sigmoid'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# ENSEMBLE NETWORK
def create_model():
    # LSTM branch
    input_lstm = Input(shape=(586, 1), name='input_lstm')
    x = LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)(input_lstm)
    x = Dropout(0.5)(x)
    x = LSTM(64, activation='tanh', recurrent_activation='sigmoid')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    
    # Dense branch
    input_dense = Input(shape=(586,), name='input_dense')
    y = Dense(128, activation='relu')(input_dense)
    y = Dropout(0.5)(y)
    y = Dense(64, activation='relu')(y)
    y = Dropout(0.5)(y)
    
    combined = concatenate([x, y])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=[input_lstm, input_dense], outputs=output)
    return model
