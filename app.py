"""
AI-Assisted Resume Portfolio Generator — Multimodal Web Front-End
==================================================================
Upload PDF, Word (.docx/.doc), Images (.png/.jpg/.webp), Text/MD documents,
or paste text directly. Processes documents using Gemini multimodal vision & AI pipeline.
"""

import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, Response, jsonify

from pipeline import (
    PortfolioError,
    extract_from_file_bytes,
    build_multimodal_contents,
    call_gemini,
    parse_portfolio_json,
    render_portfolio,
    write_resume_text,
    EMPTY_PORTFOLIO,
)

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit

SAMPLE_RESUME_TEXT = """Alex Rivera
Senior Full-Stack Engineer & AI Specialist
Email: alex.rivera@example.com | Phone: +1 (555) 019-2834
LinkedIn: https://linkedin.com/in/alexrivera-dev | GitHub: https://github.com/alexrivera-dev

SUMMARY:
Innovative Full-Stack Software Engineer with 6+ years of experience engineering scalable web applications, real-time microservices, and AI-driven solutions. Proven track record in modernizing legacy architectures and deploying machine learning pipelines in cloud environments.

SKILLS:
- Languages: Python, JavaScript, TypeScript, Go, HTML5, CSS3, SQL
- Frameworks: React, Next.js, Node.js, Flask, FastAPI, TailwindCSS
- Cloud & AI: AWS (Lambda, S3), Docker, Kubernetes, Google Gemini API, PyTorch
- Databases & Tools: PostgreSQL, MongoDB, Redis, Git, CI/CD pipelines, Jest, PyTest

EXPERIENCE:
Senior Full-Stack Engineer @ Apex Tech Solutions (2022 - Present)
- Architected and released an enterprise analytics dashboard serving 150,000+ daily active users, reducing query latency by 45%.
- Integrated generative AI capabilities into customer support workflows, handling over 25,000 automated queries monthly.
- Mentored junior engineers and led code quality audits across cross-functional teams.

Software Engineer @ Quantum Byte Labs (2019 - 2022)
- Built high-performance RESTful and GraphQL APIs powering mobile and web client platforms.
- Containerized microservices using Docker & Kubernetes, achieving 99.98% uptime.

EDUCATION:
B.S. in Computer Science @ University of California, Berkeley (2015 - 2019)
- Focus: Software Engineering & Data Structures. Magna Cum Laude.

PROJECTS:
Portfolio AI Generator
- Automated portfolio web application built with Python Flask, Jinja2, and Google Gemini API.
- Converts raw resume documents (PDF, Word, Images) into responsive, high-performance web pages.

Smart Task Flow
- Real-time collaborative task management app built with React, Node.js, and WebSockets.

ACHIEVEMENTS:
- Winner of San Francisco AI Hackathon 2024 (Best Developer Tool category).
- Published open-source Python library with over 1,200 GitHub stars.
"""


@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")


@app.route("/api/sample_resume", methods=["GET"])
def sample_resume():
    return jsonify({"text": SAMPLE_RESUME_TEXT})


@app.route("/api/generate_json", methods=["POST"])
def generate_json_api():
    try:
        api_key = request.form.get("api_key", "").strip()
        model = request.form.get("model", "gemini-2.5-flash").strip()
        user_text = request.form.get("resume_text", "").strip()

        files = request.files.getlist("resume_files")
        if not files or (len(files) == 1 and files[0].filename == ""):
            legacy_file = request.files.get("resume_pdf")
            files = [legacy_file] if legacy_file and legacy_file.filename else []

        extracted_items = []
        combined_text_for_file = []

        if user_text:
            combined_text_for_file.append(user_text)

        for file in files:
            if not file or not file.filename:
                continue
            file_bytes = file.read()
            if not file_bytes:
                continue
            item = extract_from_file_bytes(file_bytes, file.filename, file.mimetype or "")
            extracted_items.append(item)
            if item["type"] == "text":
                combined_text_for_file.append(item["content"])

        if not extracted_items and not user_text:
            raise PortfolioError("Please upload at least one file (PDF, Word, Image, Text) or paste resume text.")

        if combined_text_for_file:
            write_resume_text("\n\n".join(combined_text_for_file))

        contents = build_multimodal_contents(extracted_items, extra_text=user_text)
        raw_json = call_gemini(contents, api_key=api_key, model_name=model)
        portfolio_data = parse_portfolio_json(raw_json)

        return jsonify({"success": True, "data": portfolio_data})
    except PortfolioError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": f"Unexpected error: {exc}"}), 500


@app.route("/api/render_html", methods=["POST"])
def render_html_api():
    try:
        req_data = request.get_json(force=True)
        portfolio_data = req_data.get("portfolio", {})
        theme = req_data.get("theme", "dark")
        html = render_portfolio(portfolio_data, theme=theme)
        return jsonify({"success": True, "html": html})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/generate", methods=["POST"])
def generate():
    try:
        api_key = request.form.get("api_key", "").strip()
        model = request.form.get("model", "gemini-2.5-flash").strip()
        theme = request.form.get("theme", "dark").strip()
        user_text = request.form.get("resume_text", "").strip()

        files = request.files.getlist("resume_files")
        if not files or (len(files) == 1 and files[0].filename == ""):
            legacy_file = request.files.get("resume_pdf")
            files = [legacy_file] if legacy_file and legacy_file.filename else []

        extracted_items = []
        combined_text_for_file = []

        if user_text:
            combined_text_for_file.append(user_text)

        for file in files:
            if not file or not file.filename:
                continue
            file_bytes = file.read()
            if not file_bytes:
                continue
            item = extract_from_file_bytes(file_bytes, file.filename, file.mimetype or "")
            extracted_items.append(item)
            if item["type"] == "text":
                combined_text_for_file.append(item["content"])

        if not extracted_items and not user_text:
            return render_template("upload.html", error="Please upload a file or paste resume text."), 400

        if combined_text_for_file:
            write_resume_text("\n\n".join(combined_text_for_file))

        contents = build_multimodal_contents(extracted_items, extra_text=user_text)
        raw_json = call_gemini(contents, api_key=api_key, model_name=model)
        portfolio_data = parse_portfolio_json(raw_json)
        html = render_portfolio(portfolio_data, theme=theme)
        return Response(html, mimetype="text/html")
    except PortfolioError as exc:
        return render_template("upload.html", error=str(exc)), 400
    except Exception as exc:
        return render_template("upload.html", error=f"Error generating portfolio: {exc}"), 500


@app.errorhandler(413)
def too_large(_exc):
    return render_template("upload.html", error="Uploaded files exceed maximum allowed size (32 MB)."), 413


if __name__ == "__main__":
    app.run(debug=True, port=5000)
