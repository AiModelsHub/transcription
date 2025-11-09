"""
Conversation Summary Service

This module provides services for generating structured summaries of conversations using OpenAI.
"""

from typing import List
import json
from app.models.conversation_summary import ConversationSummary
from app.utils.prompt_templates import generate_conversation_summary_prompt
from app.core.config import settings
import openai
import google.generativeai as genai

genai.configure(api_key=settings.gemini_api_key)
openai.api_key = settings.openai_api_key

class ConversationSummaryService:
    """
    Service for generating structured conversation summaries using OpenAI.
    """

    def __init__(self, provider: str = "gemini"):
        """
        Initialize the conversation summary service.

        Args:
            provider (str): LLM provider name, e.g. "openai" or "gemini" (default).
        """
        self.model_name = "gpt-4" if provider == "gpt-4" else "gemini-pro"

    def generate_summary(self, conversation_text: str) -> ConversationSummary:
        """
        Generate a structured summary of a conversation using OpenAI.

        Args:
            conversation_text (str): The full text of the conversation.

        Returns:
            ConversationSummary: A structured summary of the conversation.

        Raises:
            ValueError: If the OpenAI response cannot be parsed or is missing required fields.
        """
        prompt = generate_conversation_summary_prompt(conversation_text)

        if self.model_name == "gpt-4":
            return self.__generate_open_ai_response(prompt=prompt)
        
        return self.__generate_gemini_response(prompt=prompt)


    def get_conversation_points(self, conversation_text: str) -> List[str]:
        """
        Extract just the user demands points from a conversation using OpenAI.

        Args:
            conversation_text (str): The full text of the conversation.

        Returns:
            List[str]: List of user demands/points raised in the conversation.
        """
        summary = self.generate_summary(conversation_text)
        return summary.user_demands_points

    def __generate_open_ai_response(self, prompt: str):
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in conversation analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
            )
            
            content = response.choices[0].message.content.strip()
            
            summary_data = json.loads(content)
            
            return ConversationSummary(
                title=summary_data["title"],
                summary=summary_data["summary"],
                user_demands_points=summary_data["user_demands_points"]
            )
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse OpenAI response: {str(e)}")
        except KeyError as e:
            raise ValueError(f"Missing required field in response: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to generate summary: {str(e)}")
        
    def __generate_gemini_response(self, prompt: str):
        """
        Gemini summary generation stub.
        TODO: connect to Google Gemini API here.
        """
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            content = response.text.strip()
            summary_data = json.loads(content)
            return ConversationSummary(
                title=summary_data["title"],
                summary=summary_data["summary"],
                user_demands_points=summary_data["user_demands_points"]
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Gemini response: {str(e)}")
        except KeyError as e:
            raise ValueError(f"Missing required field in Gemini response: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to generate Gemini summary: {str(e)}")