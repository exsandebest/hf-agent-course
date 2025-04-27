from typing import Any, List

import pytesseract
from PIL import Image
from smolagents import (
    DuckDuckGoSearchTool,
    PythonInterpreterTool,
    SpeechToTextTool,
    Tool,
    VisitWebpageTool,
    WikipediaSearchTool,
)
from youtube_transcript_api import YouTubeTranscriptApi


class YouTubeTranscriptionTool(Tool):
    """
    Tool to fetch the transcript of a YouTube video given its URL.

    Args:
        video_url (str): YouTube video URL.

    Returns:
        str: Transcript of the video as a single string.
    """

    name = "youtube_transcription"
    description = "Fetches the transcript of a YouTube video given its URL"
    inputs = {
        "video_url": {"type": "string", "description": "YouTube video URL"},
    }
    output_type = "string"

    def forward(self, video_url: str) -> str:
        video_id = video_url.strip().split("v=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry["text"] for entry in transcript])


class ReadFileTool(Tool):
    """
    Tool to read a file and return its content.

    Args:
        file_path (str): Path to the file to read.

    Returns:
        str: Content of the file or error message.
    """

    name = "read_file"
    description = "Reads a file and returns its content"
    inputs = {
        "file_path": {"type": "string", "description": "Path to the file to read"},
    }
    output_type = "string"

    def forward(self, file_path: str) -> str:
        try:
            with open(file_path, "r") as file:
                return file.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"


class ExtractTextFromImageTool(Tool):
    name = "extract_text_from_image"
    description = "Extracts text from an image using pytesseract"
    inputs = {
        "image_path": {"type": "string", "description": "Path to the image file"},
    }
    output_type = "string"

    def forward(self, image_path: str) -> str:
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            return f"Error extracting text from image: {str(e)}"


def get_tools() -> List[Tool]:
    """
    Returns a list of available tools for the agent.

    Returns:
        List[Tool]: List of initialized tool instances.
    """
    tools = [
        DuckDuckGoSearchTool(),
        PythonInterpreterTool(),
        WikipediaSearchTool(),
        VisitWebpageTool(),
        SpeechToTextTool(),
        YouTubeTranscriptionTool(),
        ReadFileTool(),
        ExtractTextFromImageTool(),
    ]
    return tools
