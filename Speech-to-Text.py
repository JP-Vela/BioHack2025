import requests
import pyaudio
import wave
import time
import json
import math
from datetime import datetime
from google import genai
from collections import defaultdict
import tiktoken  # for counting tokens
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

# Google API key
client = genai.Client(api_key="AIzaSyCLQr2ZdlBzk5odjnw6nnrptxNWtq9qHRM")

class TranscriptionManager:
    def __init__(self):
        self.transcripts = defaultdict(dict)
        self.total_whisper_seconds = 0
        self.total_gemini_tokens = 0
        self.start_time = time.time()
        self.confusion_history = []  # Store (timestamp, confusion) pairs
        
    def get_simulated_confusion(self):
        """
        Generate a simulated confusion score using a sine wave 
        with a 20-second period.
        Returns a value between 0 and 1.
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Sine wave with period of 20 seconds, scaled to 0-1 range
        confusion = (math.sin(2 * math.pi * elapsed_time / 20) + 1) / 2
        return confusion
        
    def add_transcript(self, timestamp, text):
        confusion = self.get_simulated_confusion()
        
        self.transcripts[timestamp] = {
            "text": text,
            "confusion": confusion
        }
        self.confusion_history.append((timestamp, confusion))
        self.total_whisper_seconds += RECORD_SECONDS
        
    def generate_summary(self):
        if not self.transcripts:
            return None
            
        # Format transcript for Gemini
        full_text = "\n".join([
            f"{timestamp}: {data['text']} (Confusion: {data['confusion']:.2f})" 
            for timestamp, data in sorted(self.transcripts.items())
        ])
        
        # Generate main summary using Gemini with HTML formatting
        summary_prompt = """
        Analyze this conversation and generate an HTML-formatted summary with the following structure:
        1. A brief overview 
        2. A bulleted list of key points
        3. Any notable quotes
        4. Generate an outline of what was said
        
        Format your response as HTML using <p>, <ul>, <li>, and <blockquote> tags.
        
        Conversation transcript:
        {text}
        """.format(text=full_text)
        
        response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            contents=summary_prompt
        )
        
        # Generate missed content summary with HTML formatting
        missed_content_prompt = """
        Below is a conversation with confusion scores (0-1, where 1 is most confused).
        Highlight segments where confusion scores are above 0.7 and generate an HTML-formatted report with:
        1. A list of potentially missed or unclear points
        2. Suggested clarifications or follow-up questions
        3. Context for why these points might have been confusing
        
        Use appropriate HTML tags (<h3>, <p>, <ul>, <li>, etc.) for formatting.
        
        Conversation:
        {text}
        """.format(text=full_text)
        
        missed_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=missed_content_prompt
        )
        
        # Count tokens in prompts and responses
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        prompt_tokens = len(enc.encode(summary_prompt)) + len(enc.encode(missed_content_prompt))
        response_tokens = len(enc.encode(response.text)) + len(enc.encode(missed_response.text))
        self.total_gemini_tokens = prompt_tokens + response_tokens
        
        # Create the HTML report with plotly graph
        fig = make_subplots(rows=1, cols=1)
        
        # Convert timestamps to datetime objects for plotting
        timestamps = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t, _ in self.confusion_history]
        confusion_values = [c for _, c in self.confusion_history]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=confusion_values,
                mode='lines+markers',
                name='Confusion Score'
            )
        )
        
        fig.update_layout(
            title='Confusion Score Over Time',
            xaxis_title='Time',
            yaxis_title='Confusion Score',
            yaxis_range=[0, 1]
        )
        
        # Updated HTML template with better styling
        html_content = f"""
        <html>
        <head>
            <title>Transcription Summary</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px auto;
                    max-width: 1200px;
                    line-height: 1.6;
                    color: #333;
                }}
                .section {{
                    margin-bottom: 40px;
                    padding: 20px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{ color: #2c3e50; }}
                h2 {{ 
                    color: #34495e;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                }}
                .transcript {{ 
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 5px;
                    font-family: monospace;
                    white-space: pre-wrap;
                }}
                blockquote {{
                    border-left: 4px solid #3498db;
                    margin: 15px 0;
                    padding: 10px 20px;
                    background-color: #f8f9fa;
                }}
                ul {{
                    padding-left: 20px;
                }}
                li {{
                    margin-bottom: 8px;
                }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                }}
                .stat-card {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
            </style>
        </head>
        <body>
            <h1>Transcription Summary Report</h1>
            
            <div class="section">
                <h2>Summary</h2>
                {response.text}
            </div>
            
            <div class="section">
                <h2>Areas Needing Clarification</h2>
                {missed_response.text}
            </div>
            
            <div class="section">
                <h2>Confusion Score Over Time</h2>
                <div id="confusion_graph">
                    {fig.to_html(full_html=False)}
                </div>
            </div>
            
            <div class="section">
                <h2>Full Transcript</h2>
                <div class="transcript">{full_text}</div>
            </div>
            
            <div class="section">
                <h2>Usage Statistics</h2>
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value">{self.total_whisper_seconds}</div>
                        <div>Seconds Processed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{self.total_gemini_tokens}</div>
                        <div>Tokens Used</div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = "transcription_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        # Open the report in the default browser
        webbrowser.open('file://' + os.path.realpath(report_path))
        
        return {
            "summary": response.text,
            "missed_content": missed_response.text,
            "full_transcript": full_text,
            "usage_stats": {
                "whisper_audio_seconds": self.total_whisper_seconds,
                "gemini_tokens": self.total_gemini_tokens
            }
        }

def log_transcript(text, manager, filename="transcript_log.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n\n")
    manager.add_transcript(timestamp, text)

# LemonFox.ai endpoint & your API key
LEMONFOX_ENDPOINT = "https://api.lemonfox.ai/v1/audio/transcriptions"
LEMONFOX_API_KEY = "teHPLwqh1KdqnE2goFJAo8bHZaMz9guM"

# Audio recording settings
CHUNK = 1024          # Number of frames per buffer
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000          # 16 kHz sample rate is common for speech
RECORD_SECONDS = 10   # Each chunk will be 5 seconds
DEVICE_INDEX = None   # If multiple input devices, specify index here

# Prepare request headers (authorization)
headers = {
    "Authorization": f"Bearer {LEMONFOX_API_KEY}"
}

# Prepare static fields sent in the request
data = {
    "language": "english",
    "response_format": "json"
}

# Set up PyAudio
p = pyaudio.PyAudio()

# Open your microphone stream
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK
)

print(f"Recording {RECORD_SECONDS}-second chunks. Press Ctrl+C to stop/end.")

# Initialize the TranscriptionManager
manager = TranscriptionManager()

try:
    while True:
        frames = []
    
        # Record for RECORD_SECONDS
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data_chunk = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data_chunk)

        # Save chunk to temporary WAV file
        wav_filename = "temp.wav"
        wf = wave.open(wav_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
        wf.close()

        # Send the temp.wav file to LemonFox.ai
        with open(wav_filename, "rb") as audio_file:
            files = {
                "file": audio_file  # Must match the "file" field in the docs
            }
            # Make the POST request
            response = requests.post(
                LEMONFOX_ENDPOINT,
                headers=headers,
                files=files,
                data=data
            )

        # Check the response
        if response.ok:
            resp_json = response.json()
            transcript = resp_json.get("text") or resp_json
            print(transcript)
            
            # Simply log the transcript
            log_transcript(transcript, manager)
        else:
            print("Error:", response.status_code, response.text)

except KeyboardInterrupt:
    print("\nStopping... Generating summary...")
    final_summary = manager.generate_summary()
    if final_summary:
        print("\nReport generated and opened in your browser!")
        
        # Still save the JSON for reference
        with open("final_summary.json", "w") as f:
            json.dump(final_summary, f, indent=2)

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
