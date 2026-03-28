import streamlit as st

def sidebar():
    st.sidebar.title("⚙️ Settings")

    exercise = st.sidebar.selectbox(
        "Select Exercise",
        ["Pushup", "Squat"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("Upload a video to analyze form")

    return exercise