# E-Commerce ML Dashboard (HTML/CSS/JS + Flask)

This is the full web-based version of the Streamlit app, rebuilt with:
- **Frontend**: HTML, CSS, JavaScript (no framework)
- **Backend**: Flask (Python)
- **Same logic** as the original Streamlit pages

## How to run

1. Install dependencies:
   ```
   pip install flask pandas numpy scikit-learn openpyxl matplotlib seaborn
   ```

2. Start the server:
   ```
   cd ecommerce_web
   python app.py
   ```

3. Open your browser at: `http://localhost:5050`

## Pages
1. 🏠 Home — workflow overview
2. 📤 Upload — CSV/Excel upload with preview
3. 📊 Data Profiling — column classification + stats + recommendations
4. 🛠️ Preprocessing — imputation, scaling, transformation (sidebar controls)
5. 🎯 Target Configuration — choose problem type + target column
6. 📈 Visualization — bar plots, histograms, box plots
7. 🤖 Modeling — Regression, Classification, Clustering
8. 📏 Metrics — evaluation charts and scores
9. ✅ Final Summary — full project report + copy summary
