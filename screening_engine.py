"""Reusable Phase 3 ML screening logic for the FitScore AI Flask application.

This module extracts the exact Phase 3 functions from Resume_Screening_System.ipynb
so that the web application uses identical preprocessing, TF-IDF vectorization,
feature order, CatBoost Regressor scoring, and Logistic Regression classification.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from pypdf import PdfReader
from docx import Document


def load_and_verify_models(
    reg_path="resume_screening_model.pkl",
    cls_path="resume_classification_model.pkl",
):
    """Load saved Phase 2 regression and classification bundles and verify consistency."""
    if not os.path.exists(reg_path):
        raise FileNotFoundError(f"Missing regression model file: {reg_path}")
    if not os.path.exists(cls_path):
        raise FileNotFoundError(f"Missing classification model file: {cls_path}")

    reg_bundle = joblib.load(reg_path)
    cls_bundle = joblib.load(cls_path)

    required_reg_keys = {
        "model_name", "model", "tfidf_vectorizer", "feature_columns",
        "skill_synonyms", "known_skills", "edu_rank_map",
    }
    required_cls_keys = required_reg_keys | {"category_labels"}

    missing_reg = required_reg_keys - set(reg_bundle.keys())
    missing_cls = required_cls_keys - set(cls_bundle.keys())
    if missing_reg:
        raise ValueError(f"Regression bundle is missing required artifacts: {missing_reg}")
    if missing_cls:
        raise ValueError(f"Classification bundle is missing required artifacts: {missing_cls}")

    if reg_bundle["feature_columns"] != cls_bundle["feature_columns"]:
        raise ValueError("Feature column mismatch between regression and classification bundles.")

    return reg_bundle, cls_bundle


def validate_and_parse_jd(jd_text, skill_synonyms, known_skills, edu_rank_map):
    """Validate a custom job description and extract role, skills, experience, and degree."""
    if jd_text is None or not str(jd_text).strip():
        return None, "Job description cannot be empty. Please paste a valid job description."

    jd_clean = " ".join(str(jd_text).split())
    if len(jd_clean) < 20 or len(jd_clean.split()) < 5:
        return None, "Job description is too short to analyze reliably. Please include role, skills, and experience details."

    def norm_sk(skill_str):
        cleaned = skill_str.strip().lower()
        return skill_synonyms.get(cleaned, cleaned)

    # 1. Extract target role (supports custom roles beyond the 15 dataset roles)
    job_role = "Custom Role"
    role_patterns = [
        r"(?:Join us as a[n]?|hiring a[n]?|needs a[n]?|looking for a[n]?|seeking a[n]?)\s+([A-Za-z0-9\/\-\s]{2,45}?)(?:\.|,|with|to|skilled|who|for|\n)",
        r"^([A-Za-z0-9\/\-\s]{2,45}?)\s+(?:opening|role|position)\b",
    ]
    for pat in role_patterns:
        match = re.search(pat, jd_clean, flags=re.I)
        if match:
            job_role = match.group(1).strip()
            break

    # 2. Extract required and preferred skills
    req_skills = set()
    pref_skills = set()

    req_markers = [
        "Key skills:", "Required skills:", "Must have:", "Requirements:",
        "hands-on experience in", "skilled in", "You will work with",
    ]
    pref_markers = [
        "Nice to have:", "Preferred skills:", "Preferred:",
        "Familiarity with", "Bonus points for", "Bonus:", "Good to have:",
    ]

    req_section = jd_clean
    pref_section = ""
    for pm in pref_markers:
        if pm.lower() in jd_clean.lower():
            idx = jd_clean.lower().index(pm.lower())
            req_section = jd_clean[:idx]
            pref_section = jd_clean[idx + len(pm):]
            break

    for rm in req_markers:
        if rm.lower() in req_section.lower():
            idx = req_section.lower().index(rm.lower())
            raw_req = req_section[idx + len(rm):].split(".")[0]
            for item in re.split(r",|;| and ", raw_req):
                cleaned_item = item.strip()
                if cleaned_item and len(cleaned_item.split()) <= 4:
                    req_skills.add(norm_sk(cleaned_item))
            break

    if pref_section:
        raw_pref = pref_section.split(".")[0].replace("is a plus", "")
        for item in re.split(r",|;| and ", raw_pref):
            cleaned_item = item.strip()
            if cleaned_item and len(cleaned_item.split()) <= 4:
                pref_skills.add(norm_sk(cleaned_item))

    # Known-skill fallback scan across free-form text
    req_lower = req_section.lower()
    pref_lower = pref_section.lower()
    for sk in known_skills:
        pat = r"\b" + re.escape(sk.lower()) + r"\b"
        if re.search(pat, req_lower):
            req_skills.add(norm_sk(sk))
        if pref_section and re.search(pat, pref_lower):
            pref_skills.add(norm_sk(sk))

    pref_skills = pref_skills - req_skills

    # 3. Extract minimum required experience
    exp_specified = True
    if "fresher" in jd_clean.lower() or "entry-level" in jd_clean.lower() or "0-1 year" in jd_clean.lower():
        req_exp = 0
    else:
        m_exp = re.search(r"(\d+)\+?\s*(?:years?|yrs?)", jd_clean, flags=re.I)
        if m_exp:
            req_exp = int(m_exp.group(1))
        else:
            req_exp = 0
            exp_specified = False

    # 4. Extract minimum required education
    edu_specified = True
    jd_lower = jd_clean.lower()
    req_edu_label = "Not Specified"
    if any(k in jd_lower for k in ["phd", "doctorate"]):
        req_edu = edu_rank_map["PhD"]
        req_edu_label = "PhD"
    elif any(k in jd_lower for k in ["master", "m.tech", "mba", "mca", "m.sc"]):
        req_edu = edu_rank_map["Master's"]
        req_edu_label = "Master's"
    elif any(k in jd_lower for k in ["bachelor", "b.tech", "b.e.", "b.sc", "bca", "undergraduate"]):
        req_edu = edu_rank_map["Bachelor's"]
        req_edu_label = "Bachelor's"
    elif "diploma" in jd_lower:
        req_edu = edu_rank_map["Diploma"]
        req_edu_label = "Diploma"
    else:
        req_edu = 0
        edu_specified = False

    stop_kw = {
        "with", "and", "the", "for", "are", "you", "will", "work",
        "years", "experience", "degree", "skills", "required", "preferred",
    }
    jd_keywords = sorted({w for w in re.findall(r"[a-z0-9\+\#\.]{3,}", jd_lower) if w not in stop_kw})[:15]

    return {
        "jd_text": jd_clean,
        "job_role": job_role,
        "req_skills": req_skills,
        "pref_skills": pref_skills,
        "required_experience": req_exp,
        "exp_specified": exp_specified,
        "required_edu_level": req_edu,
        "required_edu_label": req_edu_label,
        "edu_specified": edu_specified,
        "keywords": jd_keywords,
    }, None


def extract_text_from_resume(file_path):
    """Extract cleaned text from a .pdf or .docx file safely."""
    if not os.path.exists(file_path):
        return None, "Uploaded file could not be read."

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in {".pdf", ".docx"}:
        return None, f"Unsupported file format '{ext}'. Only PDF and DOCX files are supported."

    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            pages_text = []
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    pages_text.append(txt)
            raw_text = "\n".join(pages_text).strip()
            if not raw_text:
                return None, "No selectable text found in PDF. If this is a scanned PDF, OCR is required."

        else:
            doc = Document(file_path)
            parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_vals = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_vals:
                        parts.append(" | ".join(row_vals))
            raw_text = "\n".join(parts).strip()
            if not raw_text:
                return None, "The DOCX document contains no readable text."

        lines = [" ".join(line.split()) for line in raw_text.replace("\r\n", "\n").split("\n")]
        cleaned_text = "\n".join([line for line in lines if line != ""]).strip()

        if len(cleaned_text.split()) < 5:
            return None, "Extracted resume text is too short (fewer than 5 words) to analyze reliably."

        return cleaned_text, None

    except Exception:
        # Never expose internal server stack traces to end users
        return None, f"Could not parse this {ext.upper().lstrip('.')} file. It may be corrupted or password-protected."


def parse_resume_experience(resume_text):
    """Extract years of experience from resume text."""
    m = re.search(
        r"(\d+)\+?\s*(?:years?|yrs?)\s+of\s+(?:professional\s+|industry\s+|relevant\s+)?experience",
        resume_text,
        flags=re.I,
    )
    if m:
        return int(m.group(1))
    durations = re.findall(r"\((\d+)\s*(?:years?|yrs?)\)", resume_text, flags=re.I)
    if durations:
        return sum(int(d) for d in durations)
    m_any = re.search(r"experience\s*[:\-]\s*(\d+)\+?\s*(?:years?|yrs?)", resume_text, flags=re.I)
    if m_any:
        return int(m_any.group(1))
    return 0


def parse_resume_education(resume_text, edu_rank_map):
    """Extract highest education tier and label from resume text."""
    low = resume_text.lower()
    if any(k in low for k in ["phd", "ph.d", "doctorate"]):
        return edu_rank_map["PhD"], "PhD"
    if any(k in low for k in ["master", "m.tech", "mca", "mba", "m.sc", "m.s."]):
        return edu_rank_map["Master's"], "Master's"
    if any(k in low for k in ["bachelor", "b.tech", "b.e.", "b.sc", "bca", "b.s."]):
        return edu_rank_map["Bachelor's"], "Bachelor's"
    if "diploma" in low:
        return edu_rank_map["Diploma"], "Diploma"
    return 0, "Not Specified"


def extract_section_bullets_from_text(resume_text, header_keywords, stop_keywords):
    """Capture lines belonging to a resume section such as PROJECTS or CERTIFICATIONS."""
    lines = resume_text.split("\n")
    capturing = False
    collected = []
    for line in lines:
        low = line.strip().lower()
        if any(low == h or low.startswith(h + ":") for h in header_keywords):
            capturing = True
            continue
        if capturing and any(low == s or low.startswith(s + ":") for s in stop_keywords):
            break
        if capturing and line.strip():
            collected.append(line.strip())
    return collected


def build_inference_features(resume_text, jd_info, reg_bundle):
    """Compute the exact 22 features used during Phase 2 training."""
    skill_synonyms = reg_bundle["skill_synonyms"]
    known_skills = reg_bundle["known_skills"]
    edu_rank_map = reg_bundle["edu_rank_map"]
    vectorizer = reg_bundle["tfidf_vectorizer"]

    def norm_sk(skill_str):
        cleaned = skill_str.strip().lower()
        return skill_synonyms.get(cleaned, cleaned)

    res_skills = set()
    m_sk = re.search(
        r"SKILLS\s*\n(.+?)(?:\n(?:EXPERIENCE|EDUCATION|PROJECTS|CERTIFICATIONS)\b|$)",
        resume_text,
        flags=re.S | re.I,
    )
    if m_sk:
        for item in re.split(r",|\||;", m_sk.group(1)):
            if item.strip() and len(item.strip().split()) <= 4:
                res_skills.add(norm_sk(item))

    lower_res = resume_text.lower()
    for sk in known_skills:
        if re.search(r"\b" + re.escape(sk.lower()) + r"\b", lower_res):
            res_skills.add(norm_sk(sk))

    req_skills = jd_info["req_skills"]
    pref_skills = jd_info["pref_skills"]

    matched_req = sorted(res_skills.intersection(req_skills))
    missing_req = sorted(req_skills.difference(res_skills))
    matched_pref = sorted(res_skills.intersection(pref_skills))

    total_req = len(req_skills)
    total_pref = len(pref_skills)
    req_pct = round((len(matched_req) / total_req * 100.0) if total_req > 0 else 0.0, 2)
    pref_pct = round((len(matched_pref) / total_pref * 100.0) if total_pref > 0 else 0.0, 2)

    cand_exp = parse_resume_experience(resume_text)
    req_exp = jd_info["required_experience"]
    exp_diff = cand_exp - req_exp
    meets_exp = 1 if cand_exp >= req_exp else 0

    header_chunk = " ".join(resume_text.split("\n")[:5]).lower()
    role_words = [w for w in re.findall(r"[a-z0-9]+", jd_info["job_role"].lower()) if len(w) > 2]
    if jd_info["job_role"] != "Custom Role" and role_words:
        matched_role_words = sum(1 for w in role_words if w in header_chunk)
        role_aligned = 1 if matched_role_words >= max(1, len(role_words) - 1) else 0
    else:
        role_aligned = 0

    cand_edu, cand_edu_label = parse_resume_education(resume_text, edu_rank_map)
    req_edu = jd_info["required_edu_level"]
    if req_edu == 0:
        meets_edu = 1 if cand_edu > 0 else 0
        edu_suit = 1.0 if cand_edu > 0 else 0.0
    elif cand_edu >= req_edu:
        meets_edu = 1
        edu_suit = 1.0
    elif cand_edu == req_edu - 1:
        meets_edu = 0
        edu_suit = 0.5
    else:
        meets_edu = 0
        edu_suit = 0.0

    proj_lines = extract_section_bullets_from_text(
        resume_text, ["projects", "key projects"], ["certifications", "education", "experience", "skills"]
    )
    cert_lines = extract_section_bullets_from_text(
        resume_text, ["certifications", "licenses & certifications"], ["projects", "education", "experience", "skills"]
    )
    num_proj = len(proj_lines)
    num_certs = len(cert_lines)

    jd_all_skills = req_skills | pref_skills
    rel_proj = 0
    for line in proj_lines:
        low_line = line.lower()
        if any(sk in low_line for sk in jd_all_skills) or any(
            canon in jd_all_skills and alias in low_line for alias, canon in skill_synonyms.items()
        ):
            rel_proj += 1
    proj_ratio = round(rel_proj / num_proj, 3) if num_proj > 0 else 0.0

    jd_tokens = set(re.findall(r"[a-z0-9\+\#\.]+", jd_info["jd_text"].lower()))
    stop_w = {"certified", "certificate", "fundamentals", "associate", "professional", "and", "in", "for", "of"}
    rel_certs = 0
    for line in cert_lines:
        c_tokens = set(re.findall(r"[a-z0-9\+\#\.]+", line.lower())) - stop_w
        if len(c_tokens.intersection(jd_tokens)) > 0:
            rel_certs += 1

    vec_res = vectorizer.transform([resume_text])
    vec_jd = vectorizer.transform([jd_info["jd_text"]])
    tfidf_sim = float(vec_res.multiply(vec_jd).sum())

    feature_dict = {
        "req_skill_match_pct": req_pct,
        "pref_skill_match_pct": pref_pct,
        "matched_req_skills_count": len(matched_req),
        "missing_req_skills_count": len(missing_req),
        "total_matched_skills": len(matched_req) + len(matched_pref),
        "total_req_skills": total_req,
        "total_pref_skills": total_pref,
        "candidate_experience": cand_exp,
        "required_experience": req_exp,
        "experience_diff": exp_diff,
        "meets_experience": meets_exp,
        "role_aligned": role_aligned,
        "candidate_edu_level": cand_edu,
        "required_edu_level": req_edu,
        "meets_education": meets_edu,
        "edu_suitability": edu_suit,
        "num_projects": num_proj,
        "relevant_projects_count": rel_proj,
        "project_relevance_ratio": proj_ratio,
        "num_certifications": num_certs,
        "relevant_certs_count": rel_certs,
        "tfidf_cosine_sim": round(tfidf_sim, 4),
    }

    details = {
        "matched_req_skills": matched_req,
        "missing_req_skills": missing_req,
        "matched_pref_skills": matched_pref,
        "cand_edu_label": cand_edu_label,
    }
    return feature_dict, details


def generate_candidate_explanation(row, jd_info):
    """Build grounded, factual explanation items from extracted features."""
    explanations = []
    req_pct = row["req_skill_match_pct"]

    if row["total_req_skills"] == 0:
        explanations.append({
            "type": "neutral",
            "text": "No explicit required skills were detected in the job description; scoring relies on text and profile features.",
        })
    elif req_pct >= 75:
        explanations.append({
            "type": "positive",
            "text": f"Strong match in required skills ({req_pct:.1f}% matched across {row['matched_req_skills_count']}/{row['total_req_skills']} skills).",
        })
    elif req_pct >= 40:
        explanations.append({
            "type": "warning",
            "text": f"Partial match in required skills ({req_pct:.1f}% matched). Missing: {', '.join(row['missing_req_skills'])}.",
        })
    else:
        explanations.append({
            "type": "negative",
            "text": f"Low required skill alignment ({req_pct:.1f}% matched). Missing key skills: {', '.join(row['missing_req_skills'])}.",
        })

    if row["total_pref_skills"] > 0:
        if row["pref_skill_match_pct"] > 0:
            explanations.append({
                "type": "positive",
                "text": f"Matches {len(row['matched_pref_skills'])} preferred skill(s) ({row['pref_skill_match_pct']:.1f}%).",
            })
        else:
            explanations.append({
                "type": "neutral",
                "text": "Does not match the job description's preferred skills.",
            })

    if jd_info["exp_specified"]:
        if row["meets_experience"] == 1:
            explanations.append({
                "type": "positive",
                "text": f"Meets the experience requirement ({row['candidate_experience']} yrs vs {row['required_experience']}+ yrs required).",
            })
        else:
            explanations.append({
                "type": "negative",
                "text": f"Below the minimum experience requirement ({row['candidate_experience']} yrs vs {row['required_experience']}+ yrs required).",
            })
    else:
        explanations.append({
            "type": "neutral",
            "text": f"Detected {row['candidate_experience']} year(s) of experience (job description did not specify a minimum).",
        })

    if jd_info["edu_specified"]:
        if row["meets_education"] == 1:
            explanations.append({
                "type": "positive",
                "text": f"Education requirement is satisfied ({row['cand_edu_label']}).",
            })
        else:
            explanations.append({
                "type": "negative",
                "text": f"Education ({row['cand_edu_label']}) is below the stated requirement ({jd_info['required_edu_label']}).",
            })

    sim_val = row["tfidf_cosine_sim"]
    if sim_val >= 0.30:
        explanations.append({"type": "positive", "text": f"Strong resume-to-JD TF-IDF text similarity ({sim_val:.2f})."})
    elif sim_val >= 0.15:
        explanations.append({"type": "neutral", "text": f"Moderate resume-to-JD TF-IDF text similarity ({sim_val:.2f})."})
    else:
        explanations.append({"type": "warning", "text": f"Low resume-to-JD TF-IDF text similarity ({sim_val:.2f})."})

    return explanations


def screen_uploaded_resumes(jd_text, file_entries, reg_bundle, cls_bundle):
    """Run full Phase 3 screening on a list of (display_filename, temp_file_path) pairs."""
    jd_info, jd_err = validate_and_parse_jd(
        jd_text,
        reg_bundle["skill_synonyms"],
        reg_bundle["known_skills"],
        reg_bundle["edu_rank_map"],
    )
    if jd_err:
        return None, jd_err

    if not file_entries:
        return None, "Please upload at least one PDF or DOCX resume."

    feature_cols = reg_bundle["feature_columns"]
    records = []
    failed_files = []

    for display_name, temp_path in file_entries:
        res_text, extract_err = extract_text_from_resume(temp_path)
        if extract_err:
            failed_files.append({"filename": display_name, "error": extract_err})
            continue

        try:
            feat_dict, details = build_inference_features(res_text, jd_info, reg_bundle)
            ordered_feats = {col: feat_dict[col] for col in feature_cols}
            records.append({**ordered_feats, **details, "filename": display_name})
        except Exception:
            failed_files.append({
                "filename": display_name,
                "error": "Unable to extract features or generate predictions for this resume.",
            })

    if not records:
        return {
            "summary": {
                "total_uploaded": len(file_entries),
                "processed_count": 0,
                "failed_count": len(failed_files),
                "highest_score": 0.0,
            },
            "jd_info": {
                "job_role": jd_info["job_role"],
                "req_skills": sorted(jd_info["req_skills"]),
                "pref_skills": sorted(jd_info["pref_skills"]),
                "required_experience": jd_info["required_experience"],
                "exp_specified": jd_info["exp_specified"],
                "required_edu_label": jd_info["required_edu_label"],
                "edu_specified": jd_info["edu_specified"],
                "jd_text": jd_info["jd_text"],
            },
            "candidates": [],
            "failed_files": failed_files,
            "analytics": {},
        }, None

    df_infer = pd.DataFrame(records)
    X_infer = df_infer[feature_cols]

    # Independent predictions from saved Regression and Classification models
    pred_scores = np.clip(reg_bundle["model"].predict(X_infer), 0.0, 100.0)
    pred_categories = cls_bundle["model"].predict(X_infer)

    df_infer["predicted_score"] = np.round(pred_scores, 2)
    df_infer["predicted_category"] = [str(cat) for cat in pred_categories]

    # Rank descending by predicted score (consistent tie-break: req_skill_match_pct desc, filename asc)
    df_infer = df_infer.sort_values(
        by=["predicted_score", "req_skill_match_pct", "filename"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    df_infer["rank"] = np.arange(1, len(df_infer) + 1)

    candidates = []
    for _, row in df_infer.iterrows():
        row_dict = row.to_dict()
        row_dict["rank"] = int(row_dict["rank"])
        row_dict["predicted_score"] = float(row_dict["predicted_score"])
        row_dict["req_skill_match_pct"] = float(row_dict["req_skill_match_pct"])
        row_dict["pref_skill_match_pct"] = float(row_dict["pref_skill_match_pct"])
        row_dict["tfidf_cosine_sim_pct"] = round(float(row_dict["tfidf_cosine_sim"]) * 100.0, 1)
        row_dict["experience_match_label"] = (
            f"Meets ({int(row_dict['candidate_experience'])}y >= {int(row_dict['required_experience'])}y)"
            if row_dict["meets_experience"] == 1
            else f"Below ({int(row_dict['candidate_experience'])}y < {int(row_dict['required_experience'])}y)"
        )
        row_dict["explanations"] = generate_candidate_explanation(row_dict, jd_info)
        candidates.append(row_dict)

    # Analytics computed strictly from actual screening results
    score_bins = [0, 0, 0, 0, 0]  # 0-20, 21-40, 41-60, 61-80, 81-100
    category_counts = {lbl: 0 for lbl in cls_bundle["category_labels"]}
    for c in candidates:
        sc = c["predicted_score"]
        if sc <= 20:
            score_bins[0] += 1
        elif sc <= 40:
            score_bins[1] += 1
        elif sc <= 60:
            score_bins[2] += 1
        elif sc <= 80:
            score_bins[3] += 1
        else:
            score_bins[4] += 1
        cat = c["predicted_category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    analytics = {
        "score_bin_labels": ["0–20", "21–40", "41–60", "61–80", "81–100"],
        "score_bin_counts": score_bins,
        "category_labels": list(category_counts.keys()),
        "category_counts": list(category_counts.values()),
        "candidate_names": [c["filename"] for c in candidates],
        "candidate_scores": [c["predicted_score"] for c in candidates],
        "candidate_skill_matches": [c["req_skill_match_pct"] for c in candidates],
    }

    return {
        "summary": {
            "total_uploaded": len(file_entries),
            "processed_count": len(candidates),
            "failed_count": len(failed_files),
            "highest_score": float(candidates[0]["predicted_score"]),
        },
        "jd_info": {
            "job_role": jd_info["job_role"],
            "req_skills": sorted(jd_info["req_skills"]),
            "pref_skills": sorted(jd_info["pref_skills"]),
            "required_experience": jd_info["required_experience"],
            "exp_specified": jd_info["exp_specified"],
            "required_edu_label": jd_info["required_edu_label"],
            "edu_specified": jd_info["edu_specified"],
            "jd_text": jd_info["jd_text"],
        },
        "candidates": candidates,
        "failed_files": failed_files,
        "analytics": analytics,
    }, None
