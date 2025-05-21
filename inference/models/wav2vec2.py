from transformers import AutoFeatureExtractor, TFAutoModelForAudioClassification
import os
import librosa
import numpy as np
import tensorflow as tf

class Wav2Vec2SpoofDetector:
    def __init__(self, model_name='ronanhansel/wav2vec2-vocoder-ft', feature_extractor_checkpoint="facebook/wav2vec2-base"):
        self.model = TFAutoModelForAudioClassification.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_checkpoint)
        self.class_names = ['diffwave', 'gt', 'melgan', 'parallel_wave_gan', 'wavegrad', 'wavenet', 'wavernn']
        self.label2id = {name: i for i, name in enumerate(self.class_names)}
        self.id2label = {i: name for name, i in self.label2id.items()}
        self.real_label = 'gt'  # 'gt' is considered real

    def forward(self, file_path):
        target_sr = 16000
        max_duration_s = 4.0
        max_length = int(target_sr * max_duration_s)

        try:
            audio, sr = librosa.load(file_path, sr=target_sr)
            inputs = self.feature_extractor(
                audio,
                sampling_rate=self.feature_extractor.sampling_rate,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors="np"
            )

            input_values_np = inputs["input_values"].squeeze()
            if input_values_np.ndim == 0:
                input_values_np = np.zeros(max_length)
            elif input_values_np.ndim > 1:
                input_values_np = np.squeeze(input_values_np)

            input_values = tf.constant(np.expand_dims(input_values_np, axis=0), dtype=tf.float32)

            attention_mask_np = inputs["attention_mask"].squeeze()
            if attention_mask_np.ndim == 0:
                attention_mask_np = np.ones(max_length)
            elif attention_mask_np.ndim > 1:
                attention_mask_np = np.squeeze(attention_mask_np)

            attention_mask = tf.constant(np.expand_dims(attention_mask_np, axis=0), dtype=tf.int32)

            logits = self.model(input_values, attention_mask=attention_mask).logits
            predicted_class_id = int(tf.argmax(logits, axis=-1)[0].numpy())
            predicted_label = self.id2label.get(predicted_class_id, "Unknown")

            # Return True if fake (not 'gt'), False if real ('gt')
            return predicted_label != self.real_label

        except Exception as e:
            print(f"Error classifying {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

# Example usage:
# detector = Wav2Vec2SpoofDetector()
# file_path = input("Enter the path to the audio file: ")
# is_fake = detector.forward(file_path)
# print(f"Is fake: {is_fake}")

# ...existing code...

if __name__ == "__main__":
    detector = Wav2Vec2SpoofDetector()
    file_path = 'dataset/fake/26_496_000021_000003_gen.wav'
    result = detector.forward(file_path)
    if result is None:
        print("Classification failed.")
    elif result:
        print("The audio is classified as FAKE.")
    else:
        print("The audio is classified as REAL.")
# ...existing code...