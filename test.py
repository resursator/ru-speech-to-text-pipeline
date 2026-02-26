import torch
import pathlib
import whisper

base = pathlib.Path(__file__).parent / "examples/test"
audio = []
audio.append(base / "download_14_aac_24k.m4a")
audio.append(base / "download_14_aac_32k.m4a")
audio.append(base / "download_14_aac_48k.m4a")
audio.append(base / "download_14_audio_only.m4a")
audio.append(base / "download_14_flac.flac")
audio.append(base / "download_14_mp3_32k.mp3")
audio.append(base / "download_14_mp3_48k.mp3")
audio.append(base / "download_14_mp3_64k.mp3")
audio.append(base / "download_14_opus_16k.ogg")
audio.append(base / "download_14_opus_24k.ogg")
audio.append(base / "download_14_opus_32k.ogg")
audio.append(base / "download_14_wav.wav")

def load_model():
    return whisper.load_model("turbo")

model = load_model()

for i in range(0,len(audio)):
    print("Trying " + str(audio[i].suffix) + ", the file is " + str(audio[i]))
    try:
        results = model.transcribe(str(audio[i]))
        print(results["text"])
    except Exception:
        print("Failed extension " + str(audio[i].suffix))
        del model
        torch.cuda.empty_cache()
        model = load_model()
