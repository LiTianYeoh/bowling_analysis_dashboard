# Bowling Performance Dashboard

A Streamlit app for checking bowling score data, creating player and playstyle filters, and comparing performance through pin-hit heatmaps and Monte Carlo score simulations.

## Streamlit app

**App URL:** [Open the Bowling Performance Dashboard](https://litianyeoh-bowling-analysis-dashboard-app-36jksl.streamlit.app/)

## Features

- Upload bowling data in CSV (`.csv`) or Excel (`.xlsx`) format.
- Check that the uploaded file has the required columns and that each recorded final score matches the score calculated from the frame rolls.
- Create named filters by date range, player name, and playstyle.
- Compare filter results with pin-left vs. pin-hit probability heatmaps.
- Review Monte Carlo score statistics and score-distribution curves. Each filter runs 100,000 simulations.

## Run locally

From this directory, install the dependencies and start the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually `http://localhost:8501`.

## Input data

The dashboard expects one row per bowling round. You can download `sample_data.csv` from the app's **View Required Data Format** section and use it as a template.

The file must contain these columns:

| Column(s) | Description |
| --- | --- |
| `round_id` | Round identifier |
| `date` | Date of the round |
| `name` | Player name |
| `playstyle` | Playstyle label |
| `f1b1`, `f1b2` through `f9b1`, `f9b2` | First and second ball pin counts for frames 1–9 |
| `f10b1`, `f10b2`, `f10b3` | Ball pin counts for frame 10 |
| `final_score` | Recorded final score; checked against the score calculated from the frame data |

Use the frame roll columns to record strikes and spares as follows:

- **Strike:** record `b1 = 10` and `b2 = 0`.
- **Spare:** record the first and second rolls so that `b1 + b2 = 10` (for example, `b1 = 9` and `b2 = 1`).

Extra columns are excluded from analysis. Uploads with missing required columns, no data rows, or score mismatches cannot be used for filtering and analysis until corrected.

## Project files

- `app.py` — Streamlit interface and application flow.
- `utils.py` — Data validation, score calculations, and simulation/plotting helpers.
- `sample_data.csv` — Example input file.
- `requirements.txt` — Python dependencies.
