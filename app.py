"""FitScore AI – Flask Web Application for Resume Screening and Candidate Ranking."""

import csv
import io
import json
import os
import secrets
import tempfile
import uuid

from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from screening_engine import load_and_verify_models, screen_uploaded_resumes

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(24)
# Limit total upload payload to 16 MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {".pdf", ".docx"}

# Load Phase 2 saved models once when Flask starts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REG_MODEL_PATH = os.path.join(BASE_DIR, "resume_screening_model.pkl")
CLS_MODEL_PATH = os.path.join(BASE_DIR, "resume_classification_model.pkl")

REG_BUNDLE, CLS_BUNDLE = load_and_verify_models(REG_MODEL_PATH, CLS_MODEL_PATH)

# Temporary server-side session results store (works across worker threads without DB)
SESSION_STORE_DIR = os.path.join(tempfile.gettempdir(), "fitscore_ai_sessions")
os.makedirs(SESSION_STORE_DIR, exist_ok=True)
MAX_STORED_SESSIONS = 30


def save_session_results(screening_id, results_data):
    """Save screening results JSON in a server-side temp file scoped to screening_id."""
    os.makedirs(SESSION_STORE_DIR, exist_ok=True)
    # Prune oldest session files if store exceeds MAX_STORED_SESSIONS
    existing = sorted(
        (
            os.path.join(SESSION_STORE_DIR, fname)
            for fname in os.listdir(SESSION_STORE_DIR)
            if fname.endswith(".json")
        ),
        key=os.path.getmtime,
    )
    while len(existing) >= MAX_STORED_SESSIONS:
        oldest = existing.pop(0)
        try:
            os.remove(oldest)
        except OSError:
            pass

    file_path = os.path.join(SESSION_STORE_DIR, f"{screening_id}.json")
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(results_data, fh)


def get_current_results():
    """Retrieve results belonging strictly to the current user's screening session."""
    screening_id = session.get("screening_id")
    if not screening_id:
        return None
    # Guard against malformed session IDs
    try:
        uuid.UUID(screening_id)
    except ValueError:
        return None
    file_path = os.path.join(SESSION_STORE_DIR, f"{screening_id}.json")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    flash("Uploaded files exceed the 16 MB total upload limit. Please upload smaller files.", "error")
    return redirect(url_for("screening"))


@app.route("/health")
def health():
    """Health check endpoint for Render."""
    return jsonify({"status": "ok", "reg_model": REG_BUNDLE["model_name"], "cls_model": CLS_BUNDLE["model_name"]}), 200


@app.route("/")
def index():
    """Render the FitScore AI SaaS landing page."""
    return render_template("index.html", active_page="home")


@app.route("/screening")
def screening():
    """Render the Job Description and Resume Upload page."""
    return render_template("screening.html", active_page="screening")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Process uploaded PDF/DOCX resumes against the submitted job description."""
    jd_text = request.form.get("job_description", "")
    uploaded_files = request.files.getlist("resumes")

    valid_uploads = [f for f in uploaded_files if f and f.filename and f.filename.strip()]
    if not valid_uploads:
        flash("Please upload at least one PDF or DOCX resume before analyzing.", "error")
        return render_template("screening.html", active_page="screening", jd_text=jd_text), 400

    # Store uploaded files inside a temporary directory that auto-cleans right after processing
    with tempfile.TemporaryDirectory() as temp_dir:
        file_entries = []
        used_names = set()

        for idx, file_storage in enumerate(valid_uploads):
            safe_name = secure_filename(file_storage.filename)
            if not safe_name:
                safe_name = f"resume_{idx + 1}"

            display_name = safe_name
            if display_name in used_names:
                stem, ext = os.path.splitext(safe_name)
                display_name = f"{stem}_{idx + 1}{ext}"
            used_names.add(display_name)

            temp_path = os.path.join(temp_dir, f"{idx}_{display_name}")
            file_storage.save(temp_path)
            file_entries.append((display_name, temp_path))

        results_data, error_msg = screen_uploaded_resumes(
            jd_text, file_entries, REG_BUNDLE, CLS_BUNDLE
        )

    if error_msg:
        flash(error_msg, "error")
        return render_template("screening.html", active_page="screening", jd_text=jd_text), 400

    screening_id = str(uuid.uuid4())
    save_session_results(screening_id, results_data)
    session["screening_id"] = screening_id

    return redirect(url_for("results"))


@app.route("/results")
def results():
    """Display ranked screening results and Chart.js analytics for the current session."""
    data = get_current_results()
    if not data:
        flash("No active screening results found or your session expired. Please run a new screening.", "warning")
        return redirect(url_for("screening"))

    return render_template(
        "results.html",
        active_page="results",
        summary=data["summary"],
        jd_info=data["jd_info"],
        candidates=data["candidates"],
        failed_files=data["failed_files"],
        analytics=data["analytics"],
        analytics_json=json.dumps(data["analytics"]),
        reg_model_name=REG_BUNDLE["model_name"],
        cls_model_name=CLS_BUNDLE["model_name"],
    )


@app.route("/candidate/<int:rank>")
def candidate_details(rank):
    """Display detailed matching breakdown and explanation for a ranked candidate."""
    data = get_current_results()
    if not data:
        flash("Your screening session has expired or was cleared. Please analyze resumes again.", "warning")
        return redirect(url_for("screening"))

    candidate = next((c for c in data["candidates"] if c["rank"] == rank), None)
    if not candidate:
        flash(f"Candidate with Rank #{rank} was not found in the current screening session.", "error")
        return redirect(url_for("results"))

    return render_template(
        "candidate_details.html",
        active_page="results",
        candidate=candidate,
        jd_info=data["jd_info"],
        total_candidates=len(data["candidates"]),
        reg_model_name=REG_BUNDLE["model_name"],
        cls_model_name=CLS_BUNDLE["model_name"],
    )


@app.route("/download_csv")
def download_csv():
    """Export the current session's candidate ranking table as a downloadable CSV."""
    data = get_current_results()
    if not data or not data["candidates"]:
        flash("No screening results available to export.", "warning")
        return redirect(url_for("screening"))

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Rank",
        "Candidate",
        "Predicted Suitability Score (0-100)",
        "Predicted Category (Logistic Regression)",
        "Required Skill Match (%)",
        "Preferred Skill Match (%)",
        "Candidate Experience (Years)",
        "Required Experience (Years)",
        "Experience Match",
        "Education",
        "Matched Required Skills",
        "Missing Required Skills",
        "TF-IDF Similarity",
    ])

    for c in data["candidates"]:
        writer.writerow([
            c["rank"],
            c["filename"],
            f"{c['predicted_score']:.2f}",
            c["predicted_category"],
            f"{c['req_skill_match_pct']:.1f}%",
            f"{c['pref_skill_match_pct']:.1f}%",
            int(c["candidate_experience"]),
            int(c["required_experience"]),
            c["experience_match_label"],
            c["cand_edu_label"],
            ", ".join(c["matched_req_skills"]),
            ", ".join(c["missing_req_skills"]),
            f"{c['tfidf_cosine_sim']:.4f}",
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=resume_screening_results.csv"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
