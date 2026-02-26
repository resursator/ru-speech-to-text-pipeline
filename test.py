import torch
import pathlib
from qwen_asr import Qwen3ASRModel

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
    return Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=32,
        max_new_tokens=256,
    )

model = load_model()

for i in range(0,len(audio)):
    print("Trying " + str(audio[i].suffix) + ", the file is " + str(audio[i]))
    try:
        results = model.transcribe(
            audio=str(audio[i]),
            language=None, # set "English" to force the language
        )
        # print(results[i].language)
        print(results[0].text)
    except Exception:
        print("Failed extension " + str(audio[i].suffix))
        del model
        torch.cuda.empty_cache()
        model = load_model()
