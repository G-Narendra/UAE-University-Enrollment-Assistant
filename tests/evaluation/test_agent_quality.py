import os
import sys
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(base_dir, '.env'))
sys.path.append(base_dir)

from src.agents.enrollment_agent import EnrollmentAgent
from langchain_google_genai import ChatGoogleGenerativeAI


def run_evaluation():
    print("=" * 60)
    print("UAE UNIVERSITY ENROLLMENT ASSISTANT - EVALUATION")
    print("=" * 60)

    agent = EnrollmentAgent()
    judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)

    test_cases = [
        {
            "query": "What are the requirements for Computer Science at UAEU?",
            "expected": "GPA 3.0, EmSAT Math 1250, EmSAT English 1100"
        },
        {
            "query": "I have 3.2 GPA and EmSAT Math 1300. Am I eligible for UAEU Computer Science?",
            "expected": "eligible, mention EmSAT English requirement"
        },
        {
            "query": "What documents do I need as an international student for AUS?",
            "expected": "passport, financial guarantee, attested certificates"
        },
        {
            "query": "When is the next EmSAT exam?",
            "expected": "upcoming date, registration deadline"
        },
        {
            "query": "My German grade is 1.7, what is my UAE GPA?",
            "expected": "3.3 or 3.0 GPA conversion"
        },
    ]

    passed = 0
    for i, tc in enumerate(test_cases, 1):
        agent.reset()
        print(f"\n[{i}/{len(test_cases)}] Q: {tc['query']}")
        response = agent.chat(tc["query"])
        snippet = response[:300].replace("\n", " ")

        verdict_prompt = f"""Student query: {tc['query']}
Expected concepts: {tc['expected']}
Agent response (first 300 chars): {snippet}

Does the response correctly address the expected concepts? Answer only PASS or FAIL."""

        verdict = judge_llm.invoke(verdict_prompt).content.strip().upper()
        verdict_str = "[PASS]" if "PASS" in verdict else "[FAIL]"
        if "PASS" in verdict:
            passed += 1

        print(f"  Response: {snippet[:120]}...")
        print(f"  Verdict: {verdict_str}")

    accuracy = (passed / len(test_cases)) * 100
    print(f"\n{'='*60}")
    print(f"Score: {passed}/{len(test_cases)} ({accuracy:.1f}% accuracy)")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_evaluation()
