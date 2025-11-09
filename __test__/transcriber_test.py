import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from app.services.transcriber import process_audio_url
from app.models.conversation import Conversation
from app.core.config import settings

@pytest.mark.asyncio
@patch("app.services.transcriber.download_file", new_callable=AsyncMock)
@patch("app.services.transcriber.transcribe_with_whisper")
@patch("app.services.transcriber.get_audio_duration")
@patch("app.services.transcriber.split_to_sentences")
@patch("app.services.transcriber.estimate_durations")
@patch("app.services.transcriber.label_speakers")
@patch("os.remove")
async def test_process_audio_url(
    mock_remove: AsyncMock,
    mock_label_speakers: AsyncMock,
    mock_estimate_durations: AsyncMock,
    mock_split_to_sentences: AsyncMock,
    mock_get_audio_duration: AsyncMock,
    mock_transcribe: AsyncMock,
    mock_download_file: AsyncMock
):
    mock_download_file.return_value = "/tmp/fake_audio.mp3"
    mock_transcribe.return_value = "Hello. How are you?"
    mock_get_audio_duration.return_value = 10.0
    mock_split_to_sentences.return_value = ["Hello.", "How are you?"]
    mock_estimate_durations.return_value = [5.0, 5.0]
    mock_label_speakers.return_value = ["robot", "user"]
    audio_url = settings.audio_url
    conversation = await process_audio_url(audio_url)

    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) == 2
    assert conversation.messages[0].sender == "robot"
    assert conversation.messages[1].sender == "user"
    assert conversation.messages[0].duration == 5.0
    assert conversation.messages[1].duration == 5.0
    assert conversation.messages[0].text == "Hello."
    assert conversation.messages[1].text == "How are you?"

    mock_remove.assert_called_once_with("/tmp/fake_audio.mp3")
