import streamlit as st

st.set_page_config(
    page_title="E-Commerce ML Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

pages = [
    st.Page("home.py", title="APP", icon="🏠"),
    st.Page("pages/1_Upload.py", title="Upload", icon="📤"),
    st.Page("pages/2_Data_Profiling.py", title="Data Profiling", icon="📊"),
    st.Page("pages/3_Preprocessing.py", title="Preprocessing", icon="🛠️"),
    st.Page("pages/4_Target_Configuration.py", title="Target Configuration", icon="🎯"),
    st.Page("pages/5_Visualization.py", title="Visualization", icon="📈"),
    st.Page("pages/6_Modeling.py", title="Modeling", icon="🤖"),
    st.Page("pages/7_Metrics.py", title="Metrics", icon="📏"),
    st.Page("pages/8_Final_Summary.py", title="Final Summary", icon="✅"),
]

current_page = st.navigation(pages, position="top")
current_page.run()
