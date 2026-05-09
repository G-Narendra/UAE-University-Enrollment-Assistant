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
    """List all available UAE universities and their programs."""
    return json.dumps(list_all_universities(), indent=2)


@tool
def tool_get_requirements(university_code: str, program: str) -> str:
    """Get admission requirements for a specific UAE university and program. University codes: UAEU, AUS, HCT, Khalifa."""
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
    """Check if a student is eligible for admission. Provide student GPA, test scores, and target university/program."""
    return json.dumps(check_eligibility(
        student_gpa=student_gpa,
        emsat_math=emsat_math or None,
        emsat_english=emsat_english or None,
        university_code=university_code,
        program=program,
        emsat_biology=emsat_biology or None,
        emsat_chemistry=emsat_chemistry or None,
        sat_math=sat_math or None,
        sat_total=sat_total or None
    ), indent=2)


@tool
def tool_get_document_checklist(nationality: str, university_code: str, program: str) -> str:
    """Get the required documents checklist. Nationality must be: uae_national, gcc_national, or international."""
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

SYSTEM_PROMPT = """You are an expert UAE University Enrollment Assistant. You help students navigate the complex UAE university application process.

You have access to the following tools:
- tool_list_universities: List all available universities and programs
- tool_get_requirements: Get admission requirements for a specific university + program
- tool_check_eligibility: Check if a student meets admission requirements
- tool_get_document_checklist: Get required documents based on nationality
- tool_get_emsat_info: Get EmSAT exam schedules and details
- tool_calculate_gpa: Convert grades from other systems to UAE 4.0 GPA

INSTRUCTIONS:
1. Understand what the student needs (application guidance, eligibility check, document list, etc.)
2. Use tools proactively to get accurate data — never guess requirements
3. After checking eligibility, always provide a clear action plan with numbered steps
4. If a student is not eligible, explain exactly what they need to improve
5. Always mention application deadlines
6. Be encouraging, professional, and specific
7. End with: "Is there anything else you'd like to know about your application?"
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

        # ReAct loop: up to 5 tool calls
        for _ in range(5):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                # Final answer — no more tool calls
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

        final_answer = response.content
        self.conversation_history.append(AIMessage(content=final_answer))
        return final_answer
