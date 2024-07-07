import os
import soundfile as sf
from datasets import DatasetDict, Dataset, Audio, Features, Value
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration


class preprocessor():
    def __init__(self, dataset_root, processor):
        self.dataset_root = os.path.join(os.getcwd(), dataset_root) #"dataset_wav"
        self.processor = processor

    def load_files(self):
        audio = []
        transcriptions = []

        for root, dirs, files in os.walk(self.dataset_root):
            for file in files:
                if file.endswith('.trans.txt'):
                    transcription_path = os.path.join(root, file)
                    with open(transcription_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split(maxsplit =1)
                            if len(parts) != 2:
                                continue
                            filename, transcription = parts
                            audio_path = os.path.join(root, f"{filename}.wav")
                            if os.path.exists(audio_path):                    
                                audio.append(audio_path)
                                transcriptions.append(transcription)

                                #print(f'found {audio_path}')

        if not audio or not transcriptions:
            raise ValueError("no audio fies or transcriptions found... Directory Error?")

        data = {
            'audio': audio,
            'transcription': transcriptions
        }

        return data

    def preprocess_function(self, examples):
        audio_arrays = [audio["array"] for audio in examples["audio"]]
        #sampling_rates = [audio["sampling_rate"] for audio in examples["audio"]]
        
        #if len(set(sampling_rates)) > 1:
            #raise ValueError("sampling rate error")
        
        inputs = self.processor(audio_arrays, sampling_rate=16000, padding=True, return_tensors='pt')#
        labels = self.processor.tokenizer(examples["transcription"], padding=True, return_tensors='pt').input_ids

        output = {
            "input_features": inputs["input_features"],
            "attention_mask": inputs["attention_mask"],
            "input_ids": labels,     
        }

        return output

    def preprocess(self, data):
        features = Features({
            'audio': Audio(sampling_rate=16000),
            'transcription': Value(dtype='string')
        })

        dataset = Dataset.from_dict(data, features=features)
        dataset = dataset.train_test_split(test_size=0.1)

        encoded_dataset = dataset.map(self.preprocess_function, remove_columns=["audio", "transcription"], batched=True)
        encoded_dataset.set_format(type="torch", columns=["input_ids", "input_features", "attention_mask"])#, "labels"

        return encoded_dataset



