import streamlit as st

st.set_page_config(initial_sidebar_state="collapsed")

st.title("E-Commerce Machine Learning Dashboard")

st.markdown("""
This dashboard supports a complete machine learning workflow:

1. Upload Dataset  
2. Data Profiling  
3. Preprocessing  
4. Target Configuration  
5. Visualization  
6. Modeling  
7. Metrics  
8. Final Summary  

Use the **top navigation bar** to move through the workflow.
The **left sidebar** is now reserved only for pages that need configuration controls.
""")

st.info("Start from the Upload page in the top navigation.")
