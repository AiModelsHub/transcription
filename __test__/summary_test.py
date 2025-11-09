import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from app.routes.summary import summarize_audio
from app.models.response import ResponseWrapper
from app.core.config import settings
from app.routes.summary import AudioSummaryRequest
from app.models.conversation_summary import ConversationSummary

@pytest.mark.asyncio
@patch("app.services.transcriber.download_file", new_callable=AsyncMock)
@patch("app.services.transcriber.transcribe_with_whisper")
@patch("app.services.transcriber.get_audio_duration")
@patch("app.services.transcriber.split_to_sentences")
@patch("app.services.transcriber.estimate_durations")
@patch("app.services.transcriber.label_speakers")
@patch("os.remove")
@patch("app.services.conversation_summary_service.ConversationSummaryService.generate_summary")
async def test_summarize_audio(
    mock_generate_summary: AsyncMock,
    mock_remove: AsyncMock,
    mock_label_speakers: AsyncMock,
    mock_estimate_durations: AsyncMock,
    mock_split_to_sentences: AsyncMock,
    mock_get_audio_duration: AsyncMock,
    mock_transcribe: AsyncMock,
    mock_download_file: AsyncMock
):

    mock_download_file.return_value = "/tmp/fake_audio.mp3"
    mock_transcribe.return_value = "Hello, how can I help you? I need help with my order."
    mock_get_audio_duration.return_value = 10.0
    mock_split_to_sentences.return_value = ["Hello, how can I help you?", "I need help with my order."]
    mock_estimate_durations.return_value = [5.0, 5.0]
    mock_label_speakers.return_value = ["robot", "user"]
    
    mock_summary = ConversationSummary(
        title="Customer Support Inquiry",
        summary="Customer requested assistance with their order.",
        user_demands_points=["Order assistance needed"]
    )
    mock_generate_summary.return_value = mock_summary

    request = AudioSummaryRequest(audio_url=str(settings.test_audio_url))
    response = await summarize_audio(request)

    assert isinstance(response, ResponseWrapper)
    assert response.status == 200
    assert response.success is True
    assert response.message == "Conversation summary generated successfully."
    assert response.data["title"] == "Customer Support Inquiry"
    assert response.data["summary"] == "Customer requested assistance with their order."
    assert response.data["user_demands_points"] == ["Order assistance needed"]
    
    mock_remove.assert_called_once_with("/tmp/fake_audio.mp3")
