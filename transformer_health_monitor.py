
# # Transformer Health Monitoring System using CNN
# ## Complete Pipeline: Data Processing, Model Training, Prediction & Recommendations


# ### Cell 1: Import Required Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import warnings
import joblib
import pickle

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("✓ Libraries imported successfully")
print(f"TensorFlow version: {tf.__version__}")














# %% [markdown]
# ### Cell 2: Load and Explore the Dataset





# %%
# Load the CSV file
file_path = 'transformer_trials_10000.csv'
df = pd.read_csv(file_path)

print("=== Dataset Loaded Successfully ===")
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nDataset Info:")
print(df.info())

print(f"\nStatistical Summary:")
print(df.describe())

print(f"\nMissing values:")
print(df.isnull().sum())

# %% [markdown]
# ### Cell 3: Define Health Status Labels




# %%
def determine_health_status(row):
    """
    Determine transformer health status based on:
    - Efficiency
    - Temperature
    - Power Factor
    - Current levels
    """
    score = 0
    
    # Efficiency scoring (0-40 points)
    if row['Efficiency_percent'] >= 95:
        score += 40
    elif row['Efficiency_percent'] >= 90:
        score += 30
    elif row['Efficiency_percent'] >= 85:
        score += 20
    elif row['Efficiency_percent'] >= 80:
        score += 10
    else:
        score += 0
    
    # Temperature scoring (0-30 points)
    if row['Temperature_C'] <= 85:
        score += 30
    elif row['Temperature_C'] <= 105:
        score += 20
    elif row['Temperature_C'] <= 130:
        score += 10
    else:
        score += 0
    
    # Power Factor scoring (0-20 points)
    avg_pf = (row['Primary_Power_Factor'] + row['Secondary_Power_Factor']) / 2
    if avg_pf >= 0.85:
        score += 20
    elif avg_pf >= 0.78:
        score += 15
    elif avg_pf >= 0.70:
        score += 10
    else:
        score += 0
    
    # Current balance scoring (0-10 points)
    current_ratio = row['Secondary_Current_A'] / row['Primary_Current_A'] if row['Primary_Current_A'] > 0 else 0
    if 0.2 <= current_ratio <= 0.5:
        score += 10
    elif 0.15 <= current_ratio <= 0.6:
        score += 5
    else:
        score += 0
    
    # Determine health status
    if score >= 85:
        return 'Healthy'
    elif score >= 70:
        return 'Monitor'
    elif score >= 50:
        return 'Warning'
    elif score >= 30:
        return 'Critical'
    else:
        return 'Failure'

# Apply health status determination
df['Health_Status'] = df.apply(determine_health_status, axis=1)

# Display distribution
print("=== Health Status Distribution ===")
status_counts = df['Health_Status'].value_counts()
print(status_counts)
print(f"\nPercentages:")
print(status_counts / len(df) * 100)

# Visualize distribution
plt.figure(figsize=(10, 6))
colors = {'Healthy': 'green', 'Monitor': 'yellow', 'Warning': 'orange', 'Critical': 'red', 'Failure': 'darkred'}
status_colors = [colors.get(status, 'gray') for status in status_counts.index]
status_counts.plot(kind='bar', color=status_colors)
plt.title('Transformer Health Status Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Health Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('health_status_distribution.png')
plt.show()
print("✓ Health status distribution plot saved")









# %% [markdown]
# ### Cell 4: Feature Engineering and Selection

# %%
# Select features for the model
feature_columns = [
    'Primary_Voltage_V',
    'Primary_Current_A', 
    'Primary_Power_kW',
    'Primary_Power_Factor',
    'Secondary_Voltage_V',
    'Secondary_Current_A',
    'Secondary_Power_kW',
    'Secondary_Power_Factor',
    'Temperature_C',
    'Humidity_percent'
]

# Additional engineered features
df['Voltage_Ratio'] = df['Secondary_Voltage_V'] / df['Primary_Voltage_V']
df['Current_Ratio'] = df['Secondary_Current_A'] / df['Primary_Current_A']
df['Power_Ratio'] = df['Secondary_Power_kW'] / df['Primary_Power_kW']
df['Temp_Humidity_Index'] = df['Temperature_C'] * (df['Humidity_percent'] / 100)
df['Efficiency_Deviation'] = df['Efficiency_percent'] - df.groupby('Health_Status')['Efficiency_percent'].transform('mean')

# Update feature columns
feature_columns = feature_columns + ['Voltage_Ratio', 'Current_Ratio', 'Power_Ratio', 
                                      'Temp_Humidity_Index', 'Efficiency_Deviation']

print("=== Engineered Features ===")
print(f"Total features: {len(feature_columns)}")
print(f"Features: {feature_columns}")

# Prepare features and labels
X = df[feature_columns].values
y = df['Health_Status'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print(f"\nLabel mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {i}: {label}")

print(f"\nFeatures shape: {X.shape}")
print(f"Labels shape: {y_categorical.shape}")

# %% [markdown]
# ### Cell 5: Data Preprocessing and Reshaping for CNN

# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Reshape for CNN (samples, timesteps, features)
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_val_cnn = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

print(f"\nReshaped for CNN:")
print(f"X_train_cnn shape: {X_train_cnn.shape}")
print(f"X_val_cnn shape: {X_val_cnn.shape}")
print(f"X_test_cnn shape: {X_test_cnn.shape}")
















# %% [markdown]
# ### Cell 6: Build CNN Model Architecture

# %%
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        # First Conv1D layer
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', 
               input_shape=input_shape, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Second Conv1D layer
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Third Conv1D layer
        Conv1D(filters=256, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dropout(0.4),
        
        # Dense layers
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    return model

# Create the model
input_shape = (X_train_cnn.shape[1], 1)
num_classes = len(label_encoder.classes_)

model = create_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

# Display model architecture
print("=== CNN Model Architecture ===")
model.summary()









# %% [markdown]
# ### Cell 7: Train the Model with Callbacks

# %%
# Define callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.00001,
    verbose=1
)

model_checkpoint = callbacks.ModelCheckpoint(
    'best_transformer_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train the model
print("=== Training Model ===")
history = model.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

print("\n✓ Model training completed!")










# %% [markdown]
# ### Cell 8: Visualize Training History

# %%
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy plot
axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss plot
axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision plot
if 'precision' in history.history:
    axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
    axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

# Recall plot
if 'recall' in history.history:
    axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
    axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Print best metrics
best_epoch = np.argmax(history.history['val_accuracy'])
print(f"\n=== Best Model Performance ===")
print(f"Best epoch: {best_epoch + 1}")
print(f"Best validation accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")
print(f"Best validation loss: {history.history['val_loss'][best_epoch]:.4f}")






# %% [markdown]
# ### Cell 9: Evaluate Model on Test Set

# %%
# Load best model
best_model = load_model('best_transformer_model.h5')

# Evaluate on test set
test_loss, test_accuracy, test_precision, test_recall = best_model.evaluate(X_test_cnn, y_test, verbose=0)

print("=== Model Evaluation on Test Set ===")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")

# Make predictions
y_pred_prob = best_model.predict(X_test_cnn)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Transformer Health Status', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()







# %% [markdown]
# ### Cell 10: Save the Model in .h5 Format

# %%
# Save the final model
final_model_path = 'transformer_health_monitor_model.h5'
best_model.save(final_model_path)
print(f"✓ Model saved as '{final_model_path}'")

# Also save the scaler and label encoder
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("✓ Scaler and label encoder saved")

# Save feature columns
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("✓ Feature columns saved")

# Model summary
print("\n=== Model Summary ===")
best_model.summary()

# Verify model loading
loaded_model = load_model(final_model_path)
print(f"\n✓ Model verified - can be loaded successfully")









# %% [markdown]
# ### Cell 11: Prediction and Recommendation System

# %%
def predict_transformer_health(model, scaler, label_encoder, input_data):
    """
    Predict transformer health status and provide recommendations
    """
    # Ensure input is 2D
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
    
    # Preprocess input
    input_scaled = scaler.transform(input_data)
    input_cnn = input_scaled.reshape(input_scaled.shape[0], input_scaled.shape[1], 1)
    
    # Predict
    prediction_prob = model.predict(input_cnn, verbose=0)
    predicted_class = np.argmax(prediction_prob, axis=1)[0]
    confidence = np.max(prediction_prob) * 100
    health_status = label_encoder.inverse_transform([predicted_class])[0]
    
    # Get probabilities for all classes
    probabilities = {label: prob * 100 for label, prob in zip(label_encoder.classes_, prediction_prob[0])}
    
    return health_status, confidence, probabilities

def get_recommendations(health_status, temperature, efficiency, pf):
    """
    Generate recommendations based on health status and parameters
    """
    recommendations = {
        'Healthy': {
            'action': '✅ Normal Operation',
            'maintenance': 'Continue regular monitoring. Schedule routine maintenance every 6 months.',
            'urgency': 'Low',
            'checks': ['Monitor temperature trends', 'Check oil levels quarterly', 'Regular efficiency tracking']
        },
        'Monitor': {
            'action': '⚠️ Increased Monitoring Required',
            'maintenance': 'Schedule inspection within 2 weeks. Check for abnormal sounds or vibrations.',
            'urgency': 'Medium',
            'checks': ['Daily temperature monitoring', 'Check load balancing', 'Inspect cooling system']
        },
        'Warning': {
            'action': '🔴 Warning - Immediate Attention Needed',
            'maintenance': 'Schedule maintenance within 48 hours. Reduce load if possible.',
            'urgency': 'High',
            'checks': ['Hourly temperature checks', 'Reduce load by 20%', 'Check for oil leakage', 'Inspect insulation']
        },
        'Critical': {
            'action': '🚨 CRITICAL - Emergency Action Required',
            'maintenance': 'Immediate intervention required. Consider shutting down transformer.',
            'urgency': 'Emergency',
            'checks': ['Prepare for emergency shutdown', 'Call maintenance team immediately', 
                      'Check for burning smell', 'Monitor for arc flashes']
        },
        'Failure': {
            'action': '💀 FAILURE - Transformer at Risk of Complete Failure',
            'maintenance': 'IMMEDIATE SHUTDOWN REQUIRED. Replace transformer.',
            'urgency': 'Critical Emergency',
            'checks': ['Emergency shutdown protocol', 'Isolate from power source', 
                      'Prepare replacement transformer', 'Conduct failure analysis']
        }
    }
    
    # Get base recommendations
    rec = recommendations[health_status]
    
    # Add specific recommendations based on parameters
    specific_recs = []
    if temperature > 130:
        specific_recs.append("⚠️ EXTREME TEMPERATURE: Immediate cooling required. Check cooling fans and oil pumps.")
    elif temperature > 105:
        specific_recs.append("⚠️ HIGH TEMPERATURE: Increase cooling. Reduce load if possible.")
    
    if efficiency < 85:
        specific_recs.append("⚠️ LOW EFFICIENCY: Check for internal faults, winding issues, or core losses.")
    
    if pf < 0.75:
        specific_recs.append("⚠️ LOW POWER FACTOR: Check reactive power compensation. Consider capacitor banks.")
    
    rec['specific_checks'] = specific_recs
    
    return rec

# Example prediction using first test sample
print("\n=== Example Prediction ===")
sample_idx = 0
sample_input = X_test[sample_idx]
prediction = predict_transformer_health(best_model, scaler, label_encoder, sample_input)

print(f"Predicted Health Status: {prediction[0]}")
print(f"Confidence: {prediction[1]:.2f}%")
print("\nPrediction Probabilities:")
for status, prob in prediction[2].items():
    print(f"  {status}: {prob:.2f}%")

# Get recommendations based on prediction
sample_actual = df.iloc[sample_idx]
rec = get_recommendations(prediction[0], 
                         sample_actual['Temperature_C'],
                         sample_actual['Efficiency_percent'],
                         (sample_actual['Primary_Power_Factor'] + sample_actual['Secondary_Power_Factor']) / 2)

print("\n=== Recommendations ===")
print(f"Action: {rec['action']}")
print(f"Maintenance: {rec['maintenance']}")
print(f"Urgency: {rec['urgency']}")
print("\nRecommended Checks:")
for check in rec['checks']:
    print(f"  • {check}")
if rec['specific_checks']:
    print("\nSpecific Recommendations:")
    for spec in rec['specific_checks']:
        print(f"  • {spec}")






# %% [markdown]
# ### Cell 12: Batch Prediction Function

# %%
def batch_predict(model, scaler, label_encoder, input_data):
    """
    Make predictions on multiple samples
    """
    input_scaled = scaler.transform(input_data)
    input_cnn = input_scaled.reshape(input_scaled.shape[0], input_scaled.shape[1], 1)
    
    predictions_prob = model.predict(input_cnn, verbose=0)
    predictions_class = np.argmax(predictions_prob, axis=1)
    confidence = np.max(predictions_prob, axis=1) * 100
    health_status = label_encoder.inverse_transform(predictions_class)
    
    return health_status, confidence, predictions_prob

# Test batch prediction on test set
print("\n=== Batch Prediction on Test Set (First 10 samples) ===")
batch_results = batch_predict(best_model, scaler, label_encoder, X_test[:10])
for i in range(10):
    print(f"Sample {i+1}: {batch_results[0][i]} (Confidence: {batch_results[1][i]:.2f}%)")

# %% [markdown]
# ### Cell 13: Create Interactive Dashboard Function

# %%
def create_dashboard(model, scaler, label_encoder, test_data, test_labels, num_samples=100):
    """
    Create a simple dashboard visualization
    """
    # Make predictions
    predictions, confidence, _ = batch_predict(model, scaler, label_encoder, test_data[:num_samples])
    actual_labels = label_encoder.inverse_transform(test_labels[:num_samples])
    
    # Create summary dataframe
    summary_df = pd.DataFrame({
        'Actual': actual_labels,
        'Predicted': predictions,
        'Confidence': confidence
    })
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion matrix for the batch
    from sklearn.metrics import confusion_matrix
    cm_batch = confusion_matrix(actual_labels, predictions, labels=label_encoder.classes_)
    
    sns.heatmap(cm_batch, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_,
                ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix (Test Set Sample)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # Confidence distribution
    for status in label_encoder.classes_:
        status_mask = [pred == status for pred in predictions]
        if any(status_mask):
            status_conf = confidence[status_mask]
            axes[0, 1].hist(status_conf, alpha=0.5, label=status, bins=10)
    
    axes[0, 1].set_title('Confidence Distribution by Predicted Status', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Confidence (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy by class
    accuracy_by_class = []
    for status in label_encoder.classes_:
        mask = [actual == status for actual in actual_labels]
        if any(mask):
            acc = np.mean([predictions[i] == actual_labels[i] for i in range(len(actual_labels)) if actual_labels[i] == status])
            accuracy_by_class.append(acc)
        else:
            accuracy_by_class.append(0)
    
    axes[1, 0].bar(label_encoder.classes_, accuracy_by_class, color=['green', 'yellow', 'orange', 'red', 'darkred'])
    axes[1, 0].set_title('Accuracy by Health Status', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Health Status')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Confidence vs Accuracy
    correct_mask = [predictions[i] == actual_labels[i] for i in range(len(actual_labels))]
    correct_conf = confidence[correct_mask]
    incorrect_conf = confidence[~correct_mask]
    
    axes[1, 1].boxplot([correct_conf, incorrect_conf], labels=['Correct Predictions', 'Incorrect Predictions'])
    axes[1, 1].set_title('Confidence Distribution: Correct vs Incorrect', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Confidence (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dashboard.png')
    plt.show()
    
    return summary_df

# Run dashboard on test set
print("\n=== Generating Dashboard ===")
test_actual_encoded = np.argmax(y_test, axis=1)
dashboard_results = create_dashboard(best_model, scaler, label_encoder, X_test, test_actual_encoded, num_samples=200)

print("\n✓ Dashboard generated successfully")
print(f"\nDashboard Summary (First 10 rows):")
print(dashboard_results.head(10))







# %% [markdown]
# ### Cell 14: Real-time Monitoring Simulation

# %%
def real_time_monitoring(model, scaler, label_encoder, current_readings):
    """
    Simulate real-time monitoring of transformer
    """
    print("\n" + "="*60)
    print("REAL-TIME TRANSFORMER MONITORING")
    print("="*60)
    
    # Make prediction
    health_status, confidence, probabilities = predict_transformer_health(model, scaler, label_encoder, current_readings)
    
    # Extract key parameters
    temp = current_readings[8] if len(current_readings) > 8 else 0
    efficiency = df['Efficiency_percent'].mean()  # Approximate
    
    # Get recommendations
    rec = get_recommendations(health_status, temp, efficiency, 0.8)
    
    # Display results
    print(f"\n📊 Current Transformer Status:")
    print(f"   Health Status: {health_status}")
    print(f"   Confidence: {confidence:.2f}%")
    print(f"\n📈 Prediction Probabilities:")
    for status, prob in probabilities.items():
        print(f"   {status}: {prob:.2f}%")
    
    print(f"\n🎯 Action Required:")
    print(f"   {rec['action']}")
    print(f"\n🔧 Maintenance Recommendation:")
    print(f"   {rec['maintenance']}")
    print(f"\n⚠️ Urgency Level: {rec['urgency']}")
    
    print(f"\n✅ Recommended Checks:")
    for check in rec['checks']:
        print(f"   • {check}")
    
    if rec['specific_checks']:
        print(f"\n🔔 Specific Alerts:")
        for spec in rec['specific_checks']:
            print(f"   {spec}")
    
    return health_status, confidence

# Simulate real-time monitoring with a sample reading
print("\n=== Simulating Real-time Monitoring ===")
sample_reading = X_test[5]  # Use a test sample
monitoring_result = real_time_monitoring(best_model, scaler, label_encoder, sample_reading)

# %% [markdown]
# ### Cell 15: Final Summary and Model Usage Instructions

# %%
print("\n" + "="*60)
print("TRANSFORMER HEALTH MONITORING SYSTEM - SUMMARY")
print("="*60)

print("\n📊 Model Performance Metrics:")
print(f"  • Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  • Test Precision: {test_precision*100:.2f}%")
print(f"  • Test Recall: {test_recall*100:.2f}%")
print(f"  • Test F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall)*100:.2f}%")

print("\n💾 Files Saved:")
print("  • transformer_health_monitor_model.h5 - Trained CNN model")
print("  • best_transformer_model.h5 - Best model checkpoint")
print("  • scaler.pkl - Feature scaler for preprocessing")
print("  • label_encoder.pkl - Label encoder for health status")
print("  • feature_columns.pkl - Feature column names")
print("  • training_history.png - Training plots")
print("  • confusion_matrix.png - Confusion matrix visualization")
print("  • health_status_distribution.png - Data distribution plot")
print("  • dashboard.png - Dashboard visualization")

print("\n🚀 How to Use the Model:")
print("1. Load the model: model = load_model('transformer_health_monitor_model.h5')")
print("2. Load scaler: scaler = joblib.load('scaler.pkl')")
print("3. Load labels: label_encoder = joblib.load('label_encoder.pkl')")
print("4. Load features: with open('feature_columns.pkl', 'rb') as f: features = pickle.load(f)")
print("5. Prepare input data with same features")
print("6. Use predict_transformer_health() function for predictions")

print("\n📝 Example Usage Code:")
print("""
# Single prediction
new_data = np.array([[225.5, 25.3, 5.68, 0.85, 19.2, 12.5, 0.24, 0.82, 75.0, 45.0, ...]])
health, confidence, probs = predict_transformer_health(model, scaler, label_encoder, new_data)
print(f"Health Status: {health} (Confidence: {confidence:.2f}%)")

# Batch prediction
batch_results = batch_predict(model, scaler, label_encoder, batch_data)
""")

print("\n✓ System ready for deployment!")
print("\n" + "="*60)
print("END OF TRANSFORMER HEALTH MONITORING SYSTEM")
print("="*60)

# Save complete workspace
workspace = {
    'model': best_model,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'feature_columns': feature_columns,
    'history': history.history,
    'test_accuracy': test_accuracy,
    'test_precision': test_precision,
    'test_recall': test_recall
}

with open('transformer_health_monitor_workspace.pkl', 'wb') as f:
    pickle.dump(workspace, f)
print("\n✓ Complete workspace saved as 'transformer_health_monitor_workspace.pkl'")