# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, HttpUrl
from typing import Optional

class Settings(BaseSettings):
    """
    Application configuration settings loaded from environment variables.
    This class is configured for Pydantic v2, using SettingsConfigDict.

    Attributes:
        app_name (str): The name of the application.
        openai_api_key (str): API key for OpenAI services, loaded from the environment variable 'OPENAI_API_KEY'.
                              This field is required (indicated by '...').
        audio_url (Optional[HttpUrl]): General audio URL for the application. It will attempt to load from
                                       an environment variable named 'AUDIO_URL'.
        test_audio_url (Optional[HttpUrl]): Specific URL used for testing purposes. It will explicitly
                                            load from the 'TEST_AUDIO_URL' environment variable.
        robot_profile_path (str): File path to the robot embedding profile.
    """
    app_name: str = "Voice Transcriber Service"
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    audio_url: Optional[HttpUrl] = None
    test_audio_url: Optional[HttpUrl] = Field(None, env="TEST_AUDIO_URL")
    robot_profile_path: str = "data/robot_embed.npy"
    
    resend_api_key: str = Field(..., env="RESEND_API_KEY")
    email_from: str = Field(..., env="EMAIL_FROM")
    admin_email: str = Field(..., env="ADMIN_EMAIL")
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    model_config = SettingsConfigDict(
        env_file=".env",
        extra='forbid'
    )

settings = Settings()

