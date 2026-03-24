import streamlit as st

st.set_page_config(
    page_title="E-Commerce ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("E-Commerce Machine Learning Dashboard")

st.markdown("""
This dashboard supports a complete machine learning workflow:

1. Upload Dataset  
2. Data Profiling  
3. Target Configuration  
4. Preprocessing  
5. Visualization  
6. Modeling  
7. Metrics  
8. Final Summary  

Use the sidebar to move through the workflow.
""")

st.info("Start from the Upload page in the sidebar.")