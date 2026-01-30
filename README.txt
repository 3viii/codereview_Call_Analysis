Phase I - AI-based Call Analysis (Debt Collection)
-----------------------------------------------------
This package contains a complete Phase I working submission including:
- main.py : Full pipeline (ASR stubs, diarization, NLP, scoring, report generation)
- dashboard_app.py : Flask app to display the latest analysis (uses templates/dashboard.html)
- templates/dashboard.html : Dashboard UI using Chart.js
- config_example.json : configure "use_api": "mock" / "whisper" / "google"

How to run locally:
1) Ensure Python 3.8+ installed.
2) (Optional) Create virtual environment and install Flask:
   pip install Flask
3) Run analysis to generate outputs:
   python main.py
   -> This creates outputs/analysis.json, transcript.txt, report.csv, report.html
4) Run dashboard:
   python dashboard_app.py
   -> Open http://127.0.0.1:5000 in your browser to view the report and scores

Switching to real ASR services:
- For Whisper/OpenAI: replace transcribe_with_whisper with OpenAI API calls and set use_api to "whisper" in config_example.json
- For Google STT: replace transcribe_with_google_stt with google.cloud.speech client code and set use_api to "google"

Deliverables for submission:
- The code (main.py, dashboard_app.py, templates)
- Sample outputs in outputs/ folder (created after running main.py)
- README with instructions and documentation
