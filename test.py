from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import soundfile as sf
import torch
import os

model = Speech2TextForConditionalGeneration.from_pretrained(os.path.join(os.getcwd(), "model0001"))
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

def predict(audio_path):
    speech, sampling_rate = sf.read(audio_path)

    input_features = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    input_features = {key: value.to(model.device) for key, value in input_features.items()}

    with torch.no_grad():
        generated_ids = model.generate(input_features=input_features['input_features'], attention_mask=input_features['attention_mask'])

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return transcription

audio_file_path = os.path.join(os.getcwd(), '19-198-0001.wav')
transcription = predict(audio_file_path)
print("Transcription:", transcription)
