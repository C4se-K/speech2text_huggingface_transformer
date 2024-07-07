from pydub import AudioSegment
import os
import shutil

def convert(input, output):
    for root, dirs, files in os.walk(input):
        for filename in files:

            input_path = os.path.join(root, filename)
 
            relative_path = os.path.relpath(root, input)
            output_sub_dir = os.path.join(output, relative_path)
            if not os.path.exists(output_sub_dir):
                os.makedirs(output_sub_dir)
            output_path = os.path.join(output_sub_dir, filename)

            if filename.endswith('.flac'):
                wav_path = os.path.splitext(output_path)[0] + '.wav'
                audio = AudioSegment.from_file(input_path, format='flac')
                audio.export(wav_path, format='wav')

                #print(f'Converted {input_path} to {wav_path}')
            elif filename.endswith('.txt'):
                shutil.copy(input_path, output_path)

                #print(f'Copied {input_path} to {output_path}')

dataset = os.path.join(os.getcwd(), "dataset_wav")

if not os.path.exists(dataset):
        input = os.path.join(os.getcwd(), "LibriSpeech", "train-clean-100")
        os.makedirs(dataset)

        convert(input, dataset)
        print('completed flac -> wav conversion')
else:
     print('dataset found')