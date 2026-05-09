import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(base_dir, '.env'))
sys.path.append(base_dir)

from src.tools.enrollment_tools import (
    get_university_requirements,
    list_all_universities,
    check_eligibility,
    get_document_checklist,
    get_emsat_info,
    calculate_gpa_conversion,
)


# ── Langchain tool wrappers ──────────────────────────────────────────────────

@tool
def tool_list_universities() -> str:
    """List all available UAE universities, their locations, programs, and websites. Use this when a student asks which universities are available, wants a comparison overview, or asks about 'top' universities."""
    return json.dumps(list_all_universities(), indent=2)


@tool
def tool_get_requirements(university_code: str, program: str) -> str:
    """Get detailed admission requirements (GPA, EmSAT scores, deadlines, tuition) for a specific university and program. University codes: UAEU, AUS, HCT, Khalifa. Program examples: 'Computer Science', 'Medicine', 'Business Administration', 'Engineering', 'Computer Engineering'."""
    return json.dumps(get_university_requirements(university_code, program), indent=2)


@tool
def tool_check_eligibility(
    student_gpa: float,
    university_code: str,
    program: str,
    emsat_math: int = 0,
    emsat_english: int = 0,
    emsat_biology: int = 0,
    emsat_chemistry: int = 0,
    sat_math: int = 0,
    sat_total: int = 0
) -> str:
    """Check if a student meets admission requirements. Pass the student's GPA and any test scores they mentioned. For scores not mentioned by the student, pass 0. University codes: UAEU, AUS, HCT, Khalifa. This will show which requirements are met and which are missing."""
    return json.dumps(check_eligibility(
        student_gpa=student_gpa,
        emsat_math=emsat_math if emsat_math > 0 else None,
        emsat_english=emsat_english if emsat_english > 0 else None,
        university_code=university_code,
        program=program,
        emsat_biology=emsat_biology if emsat_biology > 0 else None,
        emsat_chemistry=emsat_chemistry if emsat_chemistry > 0 else None,
        sat_math=sat_math if sat_math > 0 else None,
        sat_total=sat_total if sat_total > 0 else None
    ), indent=2)


@tool
def tool_get_document_checklist(nationality: str, university_code: str, program: str) -> str:
    """Get the complete required documents checklist for a student. Nationality options: 'uae_national' (for UAE citizens), 'gcc_national' (for Saudi/Qatar/Kuwait/Oman/Bahrain nationals), 'international' (for all other countries including India, Pakistan, UK, US, etc.). Always call this when a student mentions their nationality."""
    return json.dumps(get_document_checklist(nationality, university_code, program), indent=2)


@tool
def tool_get_emsat_info(subject: str = "") -> str:
    """Get EmSAT exam information, upcoming dates, and registration details. Subject is optional (Math, English, Biology, Chemistry, Physics)."""
    return json.dumps(get_emsat_info(subject if subject else None), indent=2)


@tool
def tool_calculate_gpa(grade_system: str, raw_grade: float) -> str:
    """Convert a grade to UAE 4.0 GPA scale. Systems: percentage, german, indian_percentage, uk_honours."""
    return json.dumps(calculate_gpa_conversion(grade_system, raw_grade), indent=2)


TOOLS = [
    tool_list_universities,
    tool_get_requirements,
    tool_check_eligibility,
    tool_get_document_checklist,
    tool_get_emsat_info,
    tool_calculate_gpa,
]

TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are an expert UAE University Enrollment Assistant. You ALWAYS use your tools to get accurate data before answering. NEVER refuse a request — always try your best with available tools.

UNIVERSITIES you know about:
- UAEU (United Arab Emirates University, Al Ain)
- AUS (American University of Sharjah)
- HCT (Higher Colleges of Technology)
- Khalifa (Khalifa University, Abu Dhabi)

CRITICAL RULES:
1. ALWAYS call the appropriate tool before answering — never guess from memory.
2. When a student asks about eligibility: call tool_check_eligibility with whatever scores they provide. If a score isn't mentioned, pass 0 for it.
3. When a student says their nationality (India = international, Pakistan = international, UAE = uae_national, Saudi/Qatar/Kuwait/Bahrain/Oman = gcc_national): immediately call tool_get_document_checklist and show the full checklist.
4. For comparisons: call tool_get_requirements for EACH university separately and compare them side-by-side.
5. For "top universities" or rankings: list all universities with their strengths from the data — don't refuse.
6. For application guidance: after checking eligibility, ALWAYS provide a numbered step-by-step action plan including deadlines.
7. If a student wants to "apply": guide them through all steps (eligibility → documents → EmSAT → deadline).
8. Be warm, encouraging, and thorough. Students are nervous — help them confidently.

WHEN STUDENT PROVIDES NATIONALITY AFTER A UNIVERSITY/PROGRAM:
→ Immediately call tool_get_document_checklist with the right nationality code and show the complete list.
   - India/Pakistan/UK/US/etc. → "international"
   - UAE → "uae_national"
   - Saudi/Qatar/Kuwait/Oman/Bahrain → "gcc_national"

ALWAYS end with a clear action plan when enough info is available. Format it as:
**Your Action Plan:**
1. Step 1...
2. Step 2...
3. Step 3...
"""


class EnrollmentAgent:
    """ReAct Agent for UAE University Enrollment using Gemini tool-calling."""

    def __init__(self):
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)
        self.llm_with_tools = llm.bind_tools(TOOLS)
        self.conversation_history = []

    def reset(self):
        """Reset conversation for a new student session."""
        self.conversation_history = []

    def chat(self, user_message: str) -> str:
        """Process a user message through the ReAct agent loop."""
        self.conversation_history.append(HumanMessage(content=user_message))

        messages = [SystemMessage(content=SYSTEM_PROMPT)] + self.conversation_history

        # ReAct loop: up to 8 tool calls to handle complex multi-step queries
        response = None
        for _ in range(8):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            # Execute tool calls
            for tc in response.tool_calls:
                tool_fn = TOOL_MAP.get(tc["name"])
                if tool_fn:
                    try:
                        result = tool_fn.invoke(tc["args"])
                    except Exception as e:
                        result = json.dumps({"error": str(e)})
                else:
                    result = json.dumps({"error": f"Tool '{tc['name']}' not found."})

                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        final_answer = response.content if response else "I encountered an error. Please try again."
        self.conversation_history.append(AIMessage(content=final_answer))
        return final_answer

