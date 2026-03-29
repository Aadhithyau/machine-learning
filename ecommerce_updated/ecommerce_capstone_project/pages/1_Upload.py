import streamlit as st

st.set_page_config(initial_sidebar_state="collapsed")
from utils.helpers import load_uploaded_file, save_to_session, missing_value_summary

st.title("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    df = load_uploaded_file(uploaded_file)

    if df is not None:
        save_to_session("data", df)
        save_to_session("uploaded_filename", uploaded_file.name)

        st.success("Dataset uploaded successfully.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.subheader("Column Names")
        st.write(df.columns.tolist())

        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Data Types")
        dtype_df = df.dtypes.reset_index()
        dtype_df.columns = ["Column", "Data Type"]
        st.dataframe(dtype_df, use_container_width=True)

        st.subheader("Missing Value Summary")
        st.dataframe(missing_value_summary(df), use_container_width=True)

    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
else:
    st.info("Please upload a dataset to continue.")