"""Gemini-based visual filtering and layer ordering.

- Character detection: does a candidate layer contain body parts from a reference?
- Layer sorting: order extracted PSD layers from top (foreground) to bottom (background).
"""

from __future__ import annotations
import os
import re
from typing import List

from google import genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv(override=True)

GEMINI_MODEL = os.getenv("GEMINI_MODEL")
if not GEMINI_MODEL:
    raise RuntimeError("GEMINI_MODEL is not set in .env file")
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env file")

CONTAINS_CHARACTER_PROMPT = """Look at the first image (reference). 
Does the second image contain any body parts or a character from the first image? 
Answer strictly with one word: yes or no."""


def _get_client() -> genai.Client:
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Add it to your .env or export it before running."
        )
    return genai.Client(api_key=API_KEY)


def contains_character(reference: Image.Image, candidate: Image.Image) -> bool:
    """Return *True* when *candidate* contains body parts / character from *reference*.

    Both images are sent to Gemini 2.5 Flash which responds with a single
    ``yes`` or ``no`` token.  Any other response is treated as an error.
    """
    client = _get_client()
    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[CONTAINS_CHARACTER_PROMPT, reference, candidate],
        )

        answer = response.text.replace("\n", "").replace("`", "").strip().lower()
        #print(f"Gemini raw answer: {answer}")

        if answer == "yes":
            return True
        if answer == "no":
            return False
        else:
            raise ValueError(f"Gemini returned unexpected answer: '{answer}'. Expected 'yes' or 'no'.")
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        client.close()

