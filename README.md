# Rising Water — ML-Based Flood Risk Prediction

A complete academic prototype that estimates a **lower** or **higher** flood-risk class from 20 environmental and infrastructure indicators. The project includes a reproducible training pipeline, four-model comparison, a saved XGBoost classifier, and a Flask web interface.

> **Safety and scope:** This is an educational machine-learning demonstration, not an official flood forecast or warning system. Its output must not be used for evacuation, emergency response, or other safety-critical decisions. Always follow official meteorological and government advisories.

## 1. What was corrected

The supplied ZIP contained a Flask template that linked to `static/css/style.css` while the CSS file was stored at `static/style.css`; the CSS also referenced a non-existent `static/images` location. This version standardizes the paths, adds the missing Flask backend, validates every field on the server, and documents the data and model workflow. The original serialized XGBoost model was not reused because it had compatibility/version warnings and its measured performance on the included dataset did not match the 96.55% claim.

## 2. Features

- Responsive web interface with 20 separate input fields (each accepts 0–10).
- Server-side validation and friendly error handling.
- Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and XGBoost comparison.
- Accuracy, precision, recall, F1-score and ROC-AUC evaluation.
- Saved XGBoost model and scaler.
- Result page with estimated high-class probability and submitted values.
- About page, health-check endpoint, and model reports.

## 3. Dataset and target definition

The included `dataset/flood.csv` has 50,000 rows and 21 columns: 20 input indicators plus `FloodProbability`.

The original target is continuous. To match the requested classification workflow, the training script derives a binary label:

- `1` (higher-risk class) when `FloodProbability >= 0.50`
- `0` (lower-risk class) when `FloodProbability < 0.50`

The 0.50 threshold is a project convention for this dataset, not a real-world flood-warning threshold. The model predicts the derived class; the displayed probability is the model's estimated probability of the higher-risk class, not a calibrated meteorological probability.

### Input features

`MonsoonIntensity`, `TopographyDrainage`, `RiverManagement`, `Deforestation`, `Urbanization`, `ClimateChange`, `DamsQuality`, `Siltation`, `AgriculturalPractices`, `Encroachments`, `IneffectiveDisasterPreparedness`, `DrainageSystems`, `CoastalVulnerability`, `Landslides`, `Watersheds`, `DeterioratingInfrastructure`, `PopulationScore`, `WetlandLoss`, `InadequatePlanning`, `PoliticalFactors`.

Each field is a dataset score on a 0–10 scale. These are not direct rainfall measurements in millimetres or cloud-visibility readings.

## 4. Project structure

```text
rising-water/
├── app.py
├── train_model.py
├── requirements.txt
├── Procfile
├── runtime.txt
├── .gitignore
├── README.md
├── dataset/
│   └── flood.csv
├── models/
│   ├── xgboost_flood_classifier.json
│   └── standard_scaler.joblib
├── reports/
│   ├── model_metrics.json
│   ├── classification_report.txt
│   ├── confusion_matrix.json
│   └── feature_names.json
├── templates/
│   ├── home.html
│   ├── result.html
│   ├── about.html
│   └── error.html
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── images/flood_bg.jpg
├── notebooks/
│   └── rising_water_eda.ipynb
├── docs/
│   ├── PROJECT_DOCUMENTATION.md
│   └── GITHUB_UPLOAD_GUIDE.md
└── tests/
    └── test_model.py
```

## 5. Requirements

- Python 3.10 or newer (Python 3.11 recommended)
- pip
- A code editor such as VS Code
- Git (for GitHub upload)

## 6. Run locally (Windows)

Open Command Prompt or the VS Code terminal in the extracted project folder.

```bash
py -3.11 -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

If the `py -3.11` command is unavailable, use `python -m venv .venv` with an installed supported Python version.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## 7. Retrain the models

The trained model is included, so the website can be started without retraining. To reproduce the benchmark:

```bash
python train_model.py
```

This reads `dataset/flood.csv`, creates a stratified 80/20 train/test split (`random_state=42`), trains all four classifiers, selects the highest-F1 model, and writes evaluation reports. The Flask app expects the selected model to be XGBoost; if a different model wins after changing the dataset or training settings, update the app's model-loading code before deployment.

## 8. Measured results on the included dataset

The benchmark below was run on the included 50,000-row file with a held-out 20% test set. Values are rounded to two decimal places.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Decision Tree | 69.05% | 70.80% | 66.98% | 68.83% |
| Random Forest | 89.75% | 90.12% | 89.75% | 89.94% |
| KNN | 86.66% | 91.05% | 81.91% | 86.24% |
| XGBoost | **93.15%** | **93.30%** | **93.28%** | **93.29%** |

Exact values and ROC-AUC are in `reports/model_metrics.json`. These are results for the dataset and split used here; they do not prove performance on live weather or geographic data. The 96.55% figure in the supplied description was not reproduced, so it is not claimed by this repository.

## 9. Web application routes

- `/` — prediction form
- `/predict` — accepts the 20 submitted values and returns a result
- `/about` — project and methodology
- `/health` — JSON health-check response

## 10. Deployment

The repository includes a `Procfile` for platforms that support Gunicorn. Install dependencies and set a strong `SECRET_KEY` environment variable. Example start command:

```bash
gunicorn app:app
```

Deployment platforms and their Python runtime support change; check the platform's current documentation before deploying. The included prototype is not designed or validated for operational emergency use.

## 11. Troubleshooting

- **ModuleNotFoundError:** Activate the virtual environment and run `pip install -r requirements.txt`.
- **Model file missing:** Confirm that `models/xgboost_flood_classifier.json` exists and that you are running commands from the project root.
- **Port already in use:** Set another port, for example `set PORT=5001` on Windows, then run `python app.py`.
- **Template or CSS missing:** Keep the folder names exactly as shown in the project structure.
- **Prediction form validation:** All 20 inputs are required and must be between 0 and 10.

## 12. Academic use

Use this project as a starting point for learning data preprocessing, classification, model evaluation, Flask integration, and GitHub version control. Cite the original dataset source in your academic report if your institution requires it; verify the dataset licence and attribution requirements before redistribution.
