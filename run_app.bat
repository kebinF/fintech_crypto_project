@echo off
cd /d "E:\python project\FINS5545\fintech_crypto_project"
E:\envs_dirs\Project\Scripts\activate
streamlit run src/pipeline/stage4_app_implementation.py --server.port 8501
pause