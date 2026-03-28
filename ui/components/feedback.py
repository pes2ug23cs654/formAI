import streamlit as st
import plotly.express as px

def show_feedback(results):
    col1, col2, col3 = st.columns(3)

    col1.metric("✅ Form Score", f"{results['score']}%")
    col2.metric("🔁 Reps Count", results['reps'])
    col3.metric("⚠️ Mistakes", len(results['mistakes']))

    st.markdown("---")

    st.subheader("⚠️ Detected Issues")
    for mistake in results["mistakes"]:
        st.error(mistake)

    st.subheader("📈 Joint Angle Graph")
    fig = px.line(results["angles"], title="Elbow Angle Over Time")
    st.plotly_chart(fig, use_container_width=True)