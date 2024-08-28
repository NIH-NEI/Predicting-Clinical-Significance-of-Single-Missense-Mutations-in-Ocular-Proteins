### MACHINE LEARNING ALGORITHMS
rf_model = RandomForestClassifier(max_depth=None, min_samples_split=5, n_estimators=100, min_samples_leaf=1, class_weight=class_weights, random_state=42)
rf_model.fit(X_train_s, y_train)

xgb_model = xgb.XGBClassifier(max_depth=None, min_samples_split=5, min_samples_leaf=1, n_estimators=400, scale_pos_weight=class_weights[1], random_state=42, use_label_encoder=False)
xgb_model.fit(X_train_s, y_train)

dt_model = DecisionTreeClassifier(max_depth=None, min_samples_split=5, min_samples_leaf=1, class_weight=class_weights, random_state=42)
dt_model.fit(X_train_s, y_train)

svm_model = SVC(kernel='linear', C=1, probability=True, class_weight=class_weights, random_state=42)
svm_model.fit(X_train_s, y_train)

# For evaluation of each ML Algorithm:
## replace svm with model of choice
y_pred_svm = svm_model.predict(X_test_s)
y_pred_proba_svm = svm_model.predict_proba(X_test_s)[:, 1]

# Calculate metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
auc_svm = roc_auc_score(y_test, y_pred_proba_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)

# Print metrics
print("SVM Accuracy:", accuracy_svm)
print("SVM Precision:", precision_svm)
print("SVM AUC:", auc_svm)
print("SVM Recall:", recall_svm)
print("SVM F1:", f1_svm)
print("SVM Report:\n", report_svm)


### NEURAL NETWORKS
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
