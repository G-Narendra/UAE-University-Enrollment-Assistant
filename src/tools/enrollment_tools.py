import json
import os
from typing import Dict, Any, Optional


_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "universities_db.json")


def _load_db() -> Dict:
    """Load universities DB fresh from disk each time (avoids stale module-level cache)."""
    with open(_DB_PATH, "r") as f:
        return json.load(f)


def get_university_requirements(university_code: str, program: str) -> Dict[str, Any]:
    """Fetch admission requirements for a specific university and program."""
    db = _load_db()
    uni_code = university_code.upper().strip()
    uni = db.get(uni_code)
    if not uni:
        return {"error": f"University '{university_code}' not found.", "available_universities": list(db.keys())}

    # 1. Exact case-insensitive match
    prog_data = None
    query = program.lower().strip()
    for prog_name, prog_info in uni["programs"].items():
        if prog_name.lower() == query:
            prog_data = (prog_name, prog_info)
            break

    # 2. Substring / fuzzy match (handles "CS" -> "Computer Science", "Computer" -> "Computer Engineering")
    if not prog_data:
        for prog_name, prog_info in uni["programs"].items():
            prog_lower = prog_name.lower()
            if query in prog_lower or prog_lower in query:
                prog_data = (prog_name, prog_info)
                break

    # 3. Word-overlap match
    if not prog_data:
        query_words = set(query.split())
        best_score, best_match = 0, None
        for prog_name, prog_info in uni["programs"].items():
            overlap = len(query_words & set(prog_name.lower().split()))
            if overlap > best_score:
                best_score, best_match = overlap, (prog_name, prog_info)
        if best_match and best_score > 0:
            prog_data = best_match

    if not prog_data:
        return {"error": f"No close match for '{program}' at {uni['name']}.",
                "available_programs": list(uni["programs"].keys())}

    prog_name, info = prog_data
    return {
        "university": uni["name"],
        "program": prog_name,
        "location": uni["location"],
        "requirements": info,
        "application_deadlines": uni["application_deadlines"],
        "website": uni["website"]
    }


def list_all_universities() -> Dict[str, Any]:
    """List all universities and their available programs."""
    db = _load_db()
    return {
        code: {"name": uni["name"], "location": uni["location"],
               "programs": list(uni["programs"].keys()), "website": uni["website"]}
        for code, uni in db.items()
    }


def check_eligibility(
    student_gpa: float,
    emsat_math: Optional[int],
    emsat_english: Optional[int],
    university_code: str,
    program: str,
    emsat_biology: Optional[int] = None,
    emsat_chemistry: Optional[int] = None,
    sat_math: Optional[int] = None,
    sat_total: Optional[int] = None
) -> Dict[str, Any]:
    """
    Check if a student meets the admission requirements.
    Returns eligibility status and list of missing requirements.
    """
    req_result = get_university_requirements(university_code, program)
    if "error" in req_result:
        return req_result

    reqs = req_result["requirements"]
    missing = []
    met = []

    # GPA check
    min_gpa = reqs.get("min_gpa", 0)
    if student_gpa >= min_gpa:
        met.append(f"GPA {student_gpa} meets minimum {min_gpa}")
    else:
        missing.append(f"GPA: You have {student_gpa}, need {min_gpa} (deficit: {min_gpa - student_gpa:.2f})")

    # EmSAT Math
    if "emsat_math" in reqs:
        req_val = reqs["emsat_math"]
        if emsat_math is None:
            missing.append(f"EmSAT Math: Not yet provided — minimum required is {req_val}")
        elif emsat_math >= req_val:
            met.append(f"EmSAT Math {emsat_math} meets minimum {req_val}")
        else:
            missing.append(f"EmSAT Math: You have {emsat_math}, need {req_val} (need {req_val - emsat_math} more points)")

    # EmSAT English
    if "emsat_english" in reqs:
        req_val = reqs["emsat_english"]
        if emsat_english is None:
            missing.append(f"EmSAT English: Not yet provided — minimum required is {req_val}")
        elif emsat_english >= req_val:
            met.append(f"EmSAT English {emsat_english} meets minimum {req_val}")
        else:
            missing.append(f"EmSAT English: You have {emsat_english}, need {req_val} (need {req_val - emsat_english} more points)")

    # EmSAT Biology / Chemistry (for Medicine)
    for subj, score in [("emsat_biology", emsat_biology), ("emsat_chemistry", emsat_chemistry)]:
        if subj in reqs:
            req_val = reqs[subj]
            if score and score >= req_val:
                met.append(f"{subj} {score} meets minimum {req_val}")
            else:
                current = score or "not provided"
                missing.append(f"{subj}: You have {current}, need {req_val}")

    # SAT checks
    if "sat_math" in reqs and sat_math:
        req_val = reqs["sat_math"]
        if sat_math >= req_val:
            met.append(f"SAT Math {sat_math} meets minimum {req_val}")
        else:
            missing.append(f"SAT Math: You have {sat_math}, need {req_val}")

    if "sat_total" in reqs and sat_total:
        req_val = reqs["sat_total"]
        if sat_total >= req_val:
            met.append(f"SAT Total {sat_total} meets minimum {req_val}")
        else:
            missing.append(f"SAT Total: You have {sat_total}, need {req_val}")

    eligible = len(missing) == 0
    return {
        "eligible": eligible,
        "university": req_result["university"],
        "program": req_result["program"],
        "requirements_met": met,
        "missing_requirements": missing,
        "application_deadlines": req_result["application_deadlines"],
        "next_steps": "Proceed to document checklist." if eligible else "Address missing requirements first."
    }


def get_document_checklist(nationality: str, university_code: str, program: str) -> Dict[str, Any]:
    """
    Returns the required documents based on nationality and program.
    nationality: "uae_national", "gcc_national", "international"
    """
    nationality = nationality.lower().strip()

    common_docs = [
        "Original High School Certificate (Tawjihi or equivalent)",
        "Official Transcripts (last 3 years)",
        "Valid Passport copy",
        "UAE Residence Visa copy (if applicable)",
        "Emirates ID copy",
        "Passport-sized photographs (4x white background)",
        "Completed application form",
        "Application fee payment receipt"
    ]

    if nationality == "uae_national":
        extra = [
            "Family Book (Khulasat Al-Qayd)",
            "UAE National ID",
            "No-Objection Certificate (NOC) from Ministry of Education (if applicable)"
        ]
    elif nationality == "gcc_national":
        extra = [
            "GCC ID or Passport",
            "Proof of GCC nationality",
            "No-Objection Certificate from home country Ministry of Education"
        ]
    else:  # international
        extra = [
            "Attested and notarized high school certificate",
            "Official English translation of all documents",
            "Financial guarantee letter (proof of funds for tuition + living)",
            "Student Visa application (after admission)",
            "Medical fitness certificate",
            "Health insurance certificate"
        ]

    # Program-specific extras
    program_docs = []
    req_result = get_university_requirements(university_code, program)
    if "error" not in req_result:
        if req_result["requirements"].get("portfolio_required"):
            program_docs.append("Architecture Portfolio (10-15 design samples)")
        if program.lower() == "medicine":
            program_docs.append("Medical fitness certificate")
            program_docs.append("Personal statement (500-800 words)")

    return {
        "nationality_type": nationality,
        "university": university_code.upper(),
        "program": program,
        "common_documents": common_docs,
        "nationality_specific_documents": extra,
        "program_specific_documents": program_docs,
        "total_documents_required": len(common_docs) + len(extra) + len(program_docs),
        "important_note": "All foreign documents must be attested by UAE Ministry of Foreign Affairs."
    }


def get_emsat_info(subject: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns EmSAT exam information, upcoming dates, and registration details.
    """
    emsat_data = {
        "description": "Emirates Standardized Test (EmSAT) — UAE's national standardized test for university admission.",
        "registration_url": "https://emsat.moe.gov.ae",
        "subjects": {
            "Math": {"code": "EMSAT-M", "max_score": 2000, "passing_for_stem": 1100},
            "English": {"code": "EMSAT-E", "max_score": 2000, "passing_general": 900},
            "Physics": {"code": "EMSAT-P", "max_score": 2000},
            "Chemistry": {"code": "EMSAT-C", "max_score": 2000},
            "Biology": {"code": "EMSAT-B", "max_score": 2000}
        },
        "upcoming_test_dates": [
            {"date": "2026-06-08", "registration_deadline": "2026-05-25", "format": "Computer-based"},
            {"date": "2026-08-10", "registration_deadline": "2026-07-28", "format": "Computer-based"},
            {"date": "2026-10-19", "registration_deadline": "2026-10-05", "format": "Computer-based"},
            {"date": "2027-01-11", "registration_deadline": "2026-12-28", "format": "Computer-based"}
        ],
        "fee": "AED 150 per subject",
        "validity": "2 years from test date",
        "test_centers": ["Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Al Ain", "Fujairah", "Ras Al Khaimah"]
    }

    if subject:
        subj_info = emsat_data["subjects"].get(subject.capitalize())
        if subj_info:
            return {"subject": subject, "info": subj_info, "upcoming_dates": emsat_data["upcoming_test_dates"]}

    return emsat_data


def calculate_gpa_conversion(grade_system: str, raw_grade: float) -> Dict[str, Any]:
    """
    Convert grades from different systems to UAE 4.0 GPA scale.
    grade_system: "percentage", "uk_honours", "us_letter", "german", "indian_percentage"
    """
    grade_system = grade_system.lower().strip()

    if grade_system == "percentage":
        if raw_grade >= 95: gpa = 4.0
        elif raw_grade >= 90: gpa = 3.7
        elif raw_grade >= 85: gpa = 3.3
        elif raw_grade >= 80: gpa = 3.0
        elif raw_grade >= 75: gpa = 2.7
        elif raw_grade >= 70: gpa = 2.3
        elif raw_grade >= 65: gpa = 2.0
        elif raw_grade >= 60: gpa = 1.7
        else: gpa = 1.0

    elif grade_system == "german":
        # German 1-6 scale (1 = best, 6 = worst)
        if raw_grade <= 1.0: gpa = 4.0
        elif raw_grade <= 1.3: gpa = 3.7
        elif raw_grade <= 1.7: gpa = 3.3
        elif raw_grade <= 2.0: gpa = 3.0
        elif raw_grade <= 2.3: gpa = 2.7
        elif raw_grade <= 2.7: gpa = 2.3
        elif raw_grade <= 3.0: gpa = 2.0
        elif raw_grade <= 3.3: gpa = 1.7
        else: gpa = 1.0

    elif grade_system == "indian_percentage":
        if raw_grade >= 90: gpa = 4.0
        elif raw_grade >= 80: gpa = 3.5
        elif raw_grade >= 70: gpa = 3.0
        elif raw_grade >= 60: gpa = 2.5
        elif raw_grade >= 50: gpa = 2.0
        else: gpa = 1.0

    elif grade_system == "uk_honours":
        # First=4.0, Upper Second=3.5, Lower Second=3.0, Third=2.0
        if raw_grade >= 70: gpa = 4.0
        elif raw_grade >= 60: gpa = 3.5
        elif raw_grade >= 50: gpa = 3.0
        elif raw_grade >= 40: gpa = 2.0
        else: gpa = 1.0
    else:
        return {"error": f"Unknown grade system '{grade_system}'", "supported_systems": ["percentage", "german", "indian_percentage", "uk_honours"]}

    return {
        "original_system": grade_system,
        "original_grade": raw_grade,
        "uae_gpa_4_scale": round(gpa, 1),
        "note": "Conversions are approximate. Final conversion is at the university's discretion.",
        "uae_grade_descriptors": {
            "4.0": "A+ / Excellent",
            "3.7": "A / Excellent",
            "3.3": "A- / Very Good",
            "3.0": "B+ / Very Good",
            "2.7": "B / Good",
            "2.3": "B- / Good",
            "2.0": "C+ / Satisfactory",
            "1.7": "C / Satisfactory"
        }
    }
