# LLaMA-3 Audio Transcription & Meeting Minutes Generator

## Project Overview
This project automates meeting minutes generation by:
1. Transcribing audio using OpenAI's Whisper-1 model.
2. Summarizing key points using LLaMA-3.1-8B-Instruct.
3. Generating structured minutes including summary, discussion points, takeaways, and action items.

## Requirements
Ensure you have the following installed:

```bash
pip install torch torchvision torchaudio transformers openai huggingface_hub accelerate bitsandbytes dotenv
```
- CUDA-Enabled GPU (optional, but recommended for faster processing)
- Python 3.8+
- Hugging Face Account (for accessing LLaMA-3 model)
- OpenAI API Key (for Whisper transcription)

## Environment Setup
Set up authentication keys using a `.env` file:

```ini
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
```

## Installation & Authentication
### Ensure PyTorch & TorchVision Are Installed
Before using TorchVision, ensure PyTorch is installed:

```python
try:
    import torch
except ImportError:
    print("Torch not found. Installing...")
    !pip install torch --index-url https://download.pytorch.org/whl/cu118
    import torch
```

### Authenticate Hugging Face & OpenAI

```python
from huggingface_hub import login
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not HF_TOKEN or not OPENAI_API_KEY:
    raise ValueError("Ensure HF_TOKEN & OPENAI_API_KEY are set in .env file")

login(HF_TOKEN)
```

## Transcribe Audio Using Whisper

```python
from openai import OpenAI

AUDIO_MODEL = "whisper-1"
AUDIO_FILE_PATH = "your_audio_file.mp3"

openai = OpenAI(api_key=OPENAI_API_KEY)

# Load and transcribe audio
with open(AUDIO_FILE_PATH, "rb") as audio_file:
    transcription = openai.audio.transcriptions.create(
        model=AUDIO_MODEL, file=audio_file, response_format="text"
    )

print("Transcription:")
print(transcription)
```

## Generate Meeting Minutes with LLaMA-3

### Prepare Input Prompt
```python
system_message = """
You are an assistant that produces minutes of meetings from transcripts,
including a summary, key discussion points, takeaways, and action items.
"""
user_prompt = f"Generate structured minutes from the transcript:\n{transcription}"
messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_prompt}]
```

### Load & Quantize LLaMA-3 Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config, token=HF_TOKEN)
```

### Generate Meeting Minutes
```python
from transformers import TextStreamer

device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
streamer = TextStreamer(tokenizer)

print("Generating response...")
outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)
response = tokenizer.decode(outputs[0])

print("Meeting Minutes:")
print(response)
```

## Final Output
The generated meeting minutes will include:
- Summary (Attendees, Location, Date)
- Key Discussion Points
- Takeaways
- Action Items with Owners

---
### Next Steps
1. Optimize model inference using `bitsandbytes`.
2. Integrate UI using Streamlit for user-friendly input/output.
3. Deploy as an API using FastAPI.

Happy Coding!

