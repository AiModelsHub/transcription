import json
from typing import List
from app.utils.prompt_templates import generate_speaker_labeling_prompt
import google.generativeai as genai
import openai

from app.core.config import settings
openai.api_key = settings.openai_api_key
genai.configure(api_key=settings.gemini_api_key)


def label_speakers(sentences: List[str], provider: str = "gemini") -> List[str]:
    """
    Classify each sentence as spoken by 'robot' or 'user' using a selected LLM provider.

    Args:
        sentences (List[str]): List of sentences from the transcript.
        provider (str): 'openai' or 'gemini'. Defaults to 'gemini'.

    Returns:
        List[str]: Corresponding speaker labels ('robot' or 'user') for each sentence.
    """
    prompt = generate_speaker_labeling_prompt(sentences)

    if provider.lower() == "openai":
        return __label_speakers_openai(prompt, len(sentences))
    return __label_speakers_gemini(prompt, len(sentences))

def __label_speakers_openai(prompt: str, length: int) -> List[str]:
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a smart assistant that labels conversation roles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        print("OpenAI speaker labeling error:", e)
        # fallback alternating labels
        return ["robot" if i % 2 == 0 else "user" for i in range(length)]


def __label_speakers_gemini(prompt: str, length: int) -> List[str]:
    """
    Classify speakers using Google Gemini.
    """
    try:
        response = genai.ChatCompletion.create(
            model="gemini-1.5",
            messages=[
                {"role": "system", "content": "You are a smart assistant that labels conversation roles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        content = response.last or response.text or ""
        return json.loads(content)
    except Exception as e:
        print("Gemini speaker labeling error:", e)
        return ["robot" if i % 2 == 0 else "user" for i in range(length)]