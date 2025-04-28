from typing import List

from smolagents import (
    DuckDuckGoSearchTool,
    PythonInterpreterTool,
    Tool,
    VisitWebpageTool,
    WikipediaSearchTool,
)

from .describe_image_tool import DescribeImageTool
from .openai_speech_to_text_tool import OpenAISpeechToTextTool
from .read_file_tool import ReadFileTool
from .youtube_transcription_tool import YouTubeTranscriptionTool


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
        OpenAISpeechToTextTool(),
        YouTubeTranscriptionTool(),
        ReadFileTool(),
        DescribeImageTool(),
    ]
    return tools
