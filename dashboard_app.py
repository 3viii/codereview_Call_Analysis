"""
Flask dashboard to visualize the latest call analysis (Phase I).
Run: python dashboard_app.py
Open: http://127.0.0.1:5000
"""

from flask import Flask, render_template, jsonify
import json, os

app = Flask(__name__, template_folder="templates")

def load_latest_report():
    path = os.path.join("outputs","analysis.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.route("/api/report")
def api_report():
    data = load_latest_report()
    if not data:
        return jsonify({"error":"No report found. Run main.py first."})
    return jsonify(data)

@app.route("/")
def index():
    data = load_latest_report()
    return render_template("dashboard.html", data=data or {})

if __name__ == "__main__":
    app.run(debug=False)
