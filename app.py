import os
import sys
import streamlit as st
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, '.env'))
sys.path.append(base_dir)

from src.agents.enrollment_agent import EnrollmentAgent


@st.cache_resource
def get_agent_v3():
    return EnrollmentAgent()


def main():
    st.set_page_config(
        page_title="UAE University Enrollment Assistant",
        page_icon="🎓",
        layout="wide"
    )

    st.title("🎓 UAE University Enrollment Assistant")
    st.markdown("*AI Agent powered by Google Gemini — Ask anything about UAE university applications*")
    st.divider()

    agent = get_agent_v3()

    # Sidebar
    with st.sidebar:
        st.header("🏛️ Supported Universities")
        unis = {
            "🎓 UAEU": "United Arab Emirates University, Al Ain",
            "🏫 AUS": "American University of Sharjah",
            "🏛️ HCT": "Higher Colleges of Technology",
            "⚡ Khalifa": "Khalifa University, Abu Dhabi"
        }
        for k, v in unis.items():
            st.caption(f"**{k}**: {v}")

        st.divider()
        st.header("💡 Try These Questions")
        examples = [
            "What are the requirements to study Computer Science at UAEU?",
            "I have a 3.2 GPA and EmSAT Math 1300, am I eligible for UAEU CS?",
            "What documents do I need as an international student for AUS?",
            "When is the next EmSAT exam?",
            "My German grade is 1.7, what is that in UAE GPA?",
            "Compare Computer Science programs at UAEU and Khalifa University",
        ]
        for ex in examples:
            if st.button(ex, key=ex):
                st.session_state.pending_message = ex

        st.divider()
        if st.button("🔄 New Conversation", type="secondary"):
            agent.reset()
            st.session_state.messages = []
            st.rerun()

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Welcome message
        welcome = ("👋 Hello! I'm your UAE University Enrollment Assistant.\n\n"
                   "I can help you with:\n"
                   "- ✅ Checking admission requirements\n"
                   "- 📊 Eligibility assessment\n"
                   "- 📋 Document checklist\n"
                   "- 📅 EmSAT exam schedules\n"
                   "- 🔢 GPA conversion\n\n"
                   "Which university or program are you interested in?")
        st.session_state.messages.append({"role": "assistant", "content": welcome})

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle sidebar example click
    if "pending_message" in st.session_state:
        prompt = st.session_state.pop("pending_message")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                response = agent.chat(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask about UAE university enrollment..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Checking requirements..."):
                response = agent.chat(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    st.caption("⚠️ Requirements data is for guidance only. Always verify with the university directly.")


if __name__ == "__main__":
    main()
