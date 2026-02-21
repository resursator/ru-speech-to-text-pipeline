import torch
import pathlib
from qwen_asr import Qwen3ASRModel

audio0 = pathlib.Path(__file__).parent / "examples/asr_en.wav"
audio1 = pathlib.Path(__file__).parent / "examples/download_14_cleaned.wav"
audio2 = pathlib.Path(__file__).parent / "examples/download_16_cleaned.wav"

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
)

results = model.transcribe(
    audio=[ str(audio0), str(audio1), str(audio2) ],
    language=None, # set "English" to force the language
)

for i in range(0,len(results)):
    print(results[i].language)
    print(results[i].text)
