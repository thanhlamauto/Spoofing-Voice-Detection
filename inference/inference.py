from transformers import AutoFeatureExtractor, TFAutoModelForAudioClassification
import os
import librosa
import numpy as np
import tensorflow as tf # Import TensorFlow

# Make sure id2label is defined from your training setup
# Example: id2label = {0: 'diffwave', 1: 'gt', ...}

loaded_model = TFAutoModelForAudioClassification.from_pretrained('ronanhansel/wav2vec2-vocoder-ft')
# Example: 7 classes (1 real + 6 vocoders)
num_labels = 7
model_checkpoint = "facebook/wav2vec2-base" # Or other variants like large

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
class_names = ['diffwave', 'gt', 'melgan', 'parallel_wave_gan', 'wavegrad', 'wavenet', 'wavernn']
label2id = {name: i for i, name in enumerate(class_names)}
id2label = {i: name for name, i in label2id.items()}

def classify_audio(model, file_path, feature_extractor, id2label):
    """Classifies an audio file using the provided TF Wav2Vec2 model."""
    target_sr = 16000
    max_duration_s = 4.0
    max_length = int(target_sr * max_duration_s)

    try:
        # 1. Load and Resample Audio
        audio, sr = librosa.load(file_path, sr=target_sr) # Ensure loading at target SR

        # 2. Extract Features
        inputs = feature_extractor(
            audio,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors="np" # Get numpy arrays first
        )

        # 3. Prepare Tensors for TF Model
        # Ensure input_values is 1D before adding batch dim, then convert to TF Tensor
        input_values_np = inputs["input_values"].squeeze() # Remove any extra dims if present
        if input_values_np.ndim == 0: # Handle potential scalar case if audio is empty/too short after processing
             input_values_np = np.zeros(max_length) # Pad if necessary
        elif input_values_np.ndim > 1:
             print(f"Warning: input_values had unexpected ndim {input_values_np.ndim}, squeezing.")
             input_values_np = np.squeeze(input_values_np)


        input_values = tf.constant(np.expand_dims(input_values_np, axis=0), dtype=tf.float32)

        # Ensure attention_mask is 1D before adding batch dim, then convert to TF Tensor
        attention_mask_np = inputs["attention_mask"].squeeze()
        if attention_mask_np.ndim == 0:
             attention_mask_np = np.ones(max_length) # Pad if necessary
        elif attention_mask_np.ndim > 1:
             print(f"Warning: attention_mask had unexpected ndim {attention_mask_np.ndim}, squeezing.")
             attention_mask_np = np.squeeze(attention_mask_np)

        attention_mask = tf.constant(np.expand_dims(attention_mask_np, axis=0), dtype=tf.int32) # TF expects int32/64

        # --- Debugging: Check Shape ---
        print(f"Shape passed to model - input_values: {input_values.shape}, attention_mask: {attention_mask.shape}")
        # Expected shape: (1, 64000) for both

        # 4. Model Inference
        # Pass tensors to the model
        logits = model(input_values, attention_mask=attention_mask).logits

        # 5. Get Prediction
        predicted_class_id = int(tf.argmax(logits, axis=-1)[0].numpy())
        predicted_label = id2label.get(predicted_class_id, "Unknown") # Use .get for safety

        return predicted_label

    except Exception as e:
        print(f"Error classifying {file_path}: {e}")
        # You might want to print the full traceback for more detailed debugging
        import traceback
        traceback.print_exc()
        return None


file_path = input("Enter the path to the audio file: ")

predicted_label = classify_audio(loaded_model, file_path, feature_extractor, id2label)
print(f"Predicted label: {predicted_label}")