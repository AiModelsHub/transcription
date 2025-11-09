import asyncio, os
from resemblyzer import VoiceEncoder
from app.models.conversation import Message, Conversation
from app.helper.labels_conversation import label_speakers
from app.helper.duration_call import get_audio_duration, estimate_durations
from app.helper.download_file import download_file
from app.helper.split_transcriptions import split_to_sentences
from app.services.get_transcriptions import transcribe_with_whisper

encoder = VoiceEncoder()

async def process_audio_url(audio_url: str) -> Conversation:
    """
    Downloads an audio file from the given URL, transcribes it using OpenAI Whisper,
    splits the transcription into sentences, estimates durations for each sentence,
    labels each sentence's speaker using OpenAI GPT, and returns a structured Conversation.

    Args:
        audio_url (str): The URL of the audio file to process.

    Returns:
        Conversation: A Conversation object containing a list of Messages with sender labels,
                      durations, and transcribed text.

    Raises:
        Exception: Propagates exceptions raised during download, transcription,
                   duration estimation, or speaker labeling.

    Notes:
        - This function performs I/O-bound operations asynchronously.
        - If speaker labeling fails or mismatches sentence count, it falls back to alternating labels.
    """
    local_path = await download_file(audio_url)

    transcript = await asyncio.to_thread(transcribe_with_whisper, local_path)

    total_duration = get_audio_duration(local_path)

    sentences = split_to_sentences(transcript)
    durations = estimate_durations(sentences, total_duration) 

    labels = label_speakers(sentences=sentences)

    if len(labels) != len(sentences):
        print("Warning: label count doesn't match sentence count. Falling back.")
        labels = ["robot" if i % 2 == 0 else "user" for i in range(len(sentences))]

    messages = [
        Message(sender=labels[i], duration=durations[i], text=sentences[i])
        for i in range(len(sentences))
        if sentences[i]
    ]

    os.remove(local_path)
    return Conversation(messages=messages)
