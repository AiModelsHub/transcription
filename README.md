# conversation-from-audio

A FastAPI-based backend service that takes a remote audio file (MP3/WAV) and returns a structured conversation with speaker labels (`robot` / `user`) and estimated durations per sentence.

> Ideal for processing call recordings into readable conversations.

---

## 🚀 Features

- 🎧 Accepts audio URL (MP3)
- 🤖 Transcribes speech using OpenAI Whisper
- 🧠 Splits into sentences with durations
- 🔈 Labels speakers (default alternating, future: Resemblyzer voice profiles)
- 📦 REST API powered by FastAPI

---

## 🧪 Live in Codespaces

You can open this repo directly in a [GitHub Codespace](https://github.com/features/codespaces) and it will run out of the box.

1. Click **Code > Codespaces > Create codespace**
2. After codespace boots, run:

```bash
cp .env.example .env
# Add your actual OpenAI key to .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

```

## 🔐 .env Setup
Copy .env.example and fill in your OpenAI API key:

```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
## 📦 Installation (Local Dev)

```bash
git clone https://github.com/RaedRdhaounia/conversation-from-audio
cd conversation-from-audio

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env

uvicorn app.main:app --reload

```
## 📡 API Endpoint

### POST /api/conversation

This endpoint processes an audio URL to transcribe the conversation and identify speakers.

Request Body:
The request expects a JSON object with the following field:

Name	    | Type	 | Description
audio_url |	string | Public URL of the MP3 or WAV file

Example Request:

```bash 
POST http://localhost:8000/api/conversation
Content-Type: application/json

{
  "audio_url": "https://example.com/audio.mp3"
}
```

Response:
```json 
{
  "status": 200,
  "success": true,
  "message": "Transcription completed successfully.",
  "data": {
    "messages": [
      {
        "sender": "robot",
        "duration": 4,
        "text": "Bonjour, je suis ici pour vous aider..."
      },
      {
        "sender": "user",
        "duration": 4,
        "text": "Merci, je voudrais réserver une activité."
      }
    ]
  }
}
```

## 📁 Project Structure

```
conversation-from-audio/
├── .env.example
├── .gitignore
├── confitest.py
├── LICENSE
├── pytest.ini
├── README.md
├── app/
│   ├── api/
│   ├── core/
│   │   └── config.py
│   ├── helper/
│   │   ├── download_file.py
│   │   ├── duration_call.py
│   │   ├── labels_conversation.py
│   │   └── split_transcriptions.py
│   ├── models/
│   │   ├── conversation.py
│   │   └── response.py
│   ├── routes/
│   │   └── routes.py
│   ├── services/
│   │   ├── get_transcriptions.py
│   │   ├── speaker.py
│   │   └── transcriber.py
│   └── main.py
├── test/
│   ├── transcriber_test.py
│   └── .pytest_cache/
└── utils/
    ├── prompt_templates.py
    └── main.py
``

## ✅ Next Improvements
- 🎙️ Real speaker detection using Resemblyzer

- 🧪 Unit tests with mock audio

- 🐳 Docker deployment

- 🔐 API key auth or JWT

## 🐍 Python SDK

The Voice Transcriber Service provides an official Python SDK for easy integration with your Python applications.

### Installation

Install the SDK using pip:

```bash
pip install voice_transcriber_sdk
```

### Basic Usage

```python
from voice_transcriber_sdk import VoiceTranscriberClient

# Initialize with your API key
client = VoiceTranscriberClient(api_key="your_api_key_here")

# Process an audio file
result = client.process_audio("https://example.com/audio.mp3")
print(f"Messages: {result.messages}")

# Generate summary
summary = client.generate_summary("Your conversation text here")
print(f"Summary: {summary.summary}")
```

### Features

- 🎧 Audio file processing
- 🤖 Conversation summarization
- 📊 Health status checking
- 🔐 API key validation

### Rate Limits

- Audio processing: 100 requests/hour
- Summary generation: 50 requests/hour
- Health checks: Unlimited

### Error Handling

The SDK raises exceptions for:
- Invalid API key
- Rate limit exceeded
- Invalid audio URL
- Service unavailable

### Documentation

For full API documentation, visit:
https://conversation-from-audio.onrender.com/docs

### Example Project

Check out the example usage in:
[sdk/python/voice_transcriber_sdk/examples/example_usage.py](https://github.com/RaedRdhaounia/conversation-from-audio/tree/main/sdk/python/voice_transcriber_sdk/examples/example_usage.py)

### Testing

The SDK includes comprehensive test coverage. Run tests with:

```bash
cd sdk/python/voice_transcriber_sdk
python -m pytest tests/
```

## development preview

[Development Preview](https://conversation-from-audio.onrender.com/)

## 👨‍💻 Maintainer

Made by @raedrdhaounia — contributions welcome!
