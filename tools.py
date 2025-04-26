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


def get_tools():
    tools = [
        DuckDuckGoSearchTool(),
        PythonInterpreterTool(),
        WikipediaSearchTool(),
        VisitWebpageTool(),
        SpeechToTextTool(),
        YouTubeTranscriptionTool(),
    ]
    return tools
