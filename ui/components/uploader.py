import streamlit as st


def video_uploader():
    """
    Component for uploading video files for exercise form analysis.
    Returns the uploaded file object or None if no file is uploaded.
    """
    st.subheader("📹 Upload Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file for form analysis",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        help="Upload a video of your exercise to analyze the form"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
        return uploaded_file
    
    return None
