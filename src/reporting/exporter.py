import os
import json
import csv
from datetime import datetime, timezone
from typing import List, Dict, Any

class Exporter:
    @staticmethod
    def save(transcript: str, turns: List[Dict], diarized: List[Dict], intent: str, entities: Dict, scores: Dict,
             sentiment: Dict = None, speech_emotion: Dict = None, out_dir: str = "outputs"):
        os.makedirs(out_dir, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).isoformat()

        # Transcript
        with open(os.path.join(out_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(transcript)

        # JSON
        analysis = {
            "timestamp": timestamp,
            "intent": intent,
            "entities": entities,
            "scores": scores,
            "sentiment": sentiment,
            "speech_emotion": speech_emotion,
            "turns": turns,
            "diarized": diarized,
        }
        with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        # CSV
        with open(os.path.join(out_dir, "report.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "call_id", "timestamp", "intent",
                "amounts", "dates", "modes",
                "listening", "communication", "persuasion", "outcome",
                "text_sentiment", "speech_emotion"
            ])
            writer.writerow([
                "call_001",
                timestamp,
                intent,
                "|".join(entities.get("Amount", [])),
                "|".join(entities.get("Date", [])),
                "|".join(entities.get("Mode", [])),
                scores.get("Listening"),
                scores.get("Communication"),
                scores.get("Persuasion"),
                scores.get("Outcome"),
                (sentiment or {}).get("label"),
                (speech_emotion or {}).get("label_pretty"),
            ])

        # HTML
        html = f"""
<html>
<head>
    <meta charset="utf-8">
    <title>Call Analysis</title>
</head>
<body>
<h1>Call Analysis Report</h1>
<p><b>Timestamp:</b> {timestamp}</p>

<h2>Intent: {intent}</h2>

<h3>Text Sentiment</h3>
<pre>{json.dumps(sentiment, ensure_ascii=False, indent=2)}</pre>

<h3>Speech Emotion (Tone)</h3>
<pre>{json.dumps(speech_emotion, ensure_ascii=False, indent=2)}</pre>

<h3>Entities</h3>
<ul>
<li>Amounts: {', '.join(entities.get('Amount', []) or ['-'])}</li>
<li>Dates: {', '.join(entities.get('Date', []) or ['-'])}</li>
<li>Modes: {', '.join(entities.get('Mode', []) or ['-'])}</li>
<li>PERSON: {', '.join(entities.get('PERSON', []) or ['-'])}</li>
<li>ORG: {', '.join(entities.get('ORG', []) or ['-'])}</li>
<li>LOC: {', '.join(entities.get('LOC', []) or ['-'])}</li>
</ul>

<h3>Scores</h3>
<ul>
<li>Listening: {scores.get('Listening')}</li>
<li>Communication: {scores.get('Communication')}</li>
<li>Persuasion: {scores.get('Persuasion')}</li>
<li>Outcome: {scores.get('Outcome')}</li>
</ul>

<h3>Transcript</h3>
<pre style="white-space: pre-wrap; font-family: monospace;">
{transcript}
</pre>
</body>
</html>
"""
        with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
            f.write(html)
