# ITAI2373 NewsBot Midterm

## Overview
This repository contains my ITAI 2373 midterm project, a NewsBot Intelligence System built in Jupyter Notebook. The project uses NLP techniques to analyze and classify news articles, including preprocessing, TF-IDF feature extraction, part-of-speech analysis, syntax analysis, sentiment analysis, named entity recognition, and multi-class classification.

## Project Files
- [Midterm_NewsBot_Intelligence_System_completed.ipynb](Midterm_NewsBot_Intelligence_System_completed.ipynb): main notebook for the assignment
- [streamlit_app.py](streamlit_app.py): bonus interactive Streamlit dashboard and live NewsBot demo
- [data/newsbot_dataset_sample.csv](data/newsbot_dataset_sample.csv): prepared sample of the dataset used in the notebook
- [outputs](outputs): saved charts, model outputs, and summary files
- [requirements.txt](requirements.txt): Python dependencies for running the project locally

## Dataset
The project uses the HuffPost News Category Dataset (`News_Category_Dataset_v3.json`) from Kaggle. For this assignment, I worked with a balanced sample across these categories:

- `POLITICS`
- `ENTERTAINMENT`
- `BUSINESS`
- `SPORTS`
- `TECH`
- `WELLNESS`

The notebook uses each article's headline and short description as the main text for analysis.

## Model Selection Note
The notebook compares Multinomial Naive Bayes, Logistic Regression, and Linear SVM. Linear SVM produced the highest Macro F1 score during evaluation, but the final interactive NewsBot uses Multinomial Naive Bayes because it also provides prediction probabilities and remained close in performance. That makes the final article-by-article output easier to interpret while still keeping the system reasonably accurate.

## Bonus Opportunities
This repository now supports two realistic extra-credit angles:

- **Interactive Dashboard (5 pts):** a Streamlit app that lets a user browse results, inspect saved charts, and run the NewsBot on new article text
- **Advanced Analysis (5 pts):** a temporal-trends view in the Streamlit app that compares article volume and sentiment by year and category

If more extra credit is needed later, the next strongest add-on would probably be a research-style model comparison or extension rather than trying to train a domain-specific NER model at the last minute.

## How To Run
1. Open a terminal in `ITAI2373-NewsBot-Midterm`.
2. Install the dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Launch Jupyter Notebook or JupyterLab.
4. Open `Midterm_NewsBot_Intelligence_System_completed.ipynb`.
5. Run the notebook from top to bottom.

To run the bonus Streamlit dashboard locally:

```bash
streamlit run streamlit_app.py
```

To host it after cloning the repo:

1. Create a free Streamlit Community Cloud account.
2. Connect your GitHub repository.
3. Set the app entrypoint to `ITAI2373-NewsBot-Midterm/streamlit_app.py` if the repo root is one level above this folder, or just `streamlit_app.py` if this folder is the repo root.
4. Deploy using the existing `requirements.txt`.

## AI Usage Overview
I used AI to help me debug and validate my work. I used AI tools like ChatGPT to help structure and plan my project, then used Codex to help me execute this plan in a local environment. I am new to running Jupyter locally, and I do not want to rely on cloud computing to get AI work done, so I used codex heavily to get a local kernel set up in VS Code and help me debug and resolve issues along the way.

## Notes
- This project was developed as a school assignment.
- I used local Jupyter instead of Google Colab for my workflow.
- The repository includes saved outputs so the analysis artifacts are easy to review.
- The Streamlit app is designed as a bonus-point companion to the notebook, not as a replacement for the written analysis.
