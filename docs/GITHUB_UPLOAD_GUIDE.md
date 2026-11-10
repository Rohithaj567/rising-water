# GitHub Upload Guide (step by step)

## A. Prepare the project
1. Extract `rising-water-complete.zip` to a folder named `rising-water`.
2. Open that folder in VS Code.
3. Check that `README.md`, `app.py`, `train_model.py`, `requirements.txt`, `dataset/`, `models/`, `templates/` and `static/` are visible.
4. Run the local installation and launch steps in the main README before uploading.

## B. Create a GitHub repository
1. Sign in to GitHub.
2. Select **New repository**.
3. Enter a repository name such as `rising-water-flood-prediction`.
4. Add a short description: `Machine-learning flood-risk classification demo using Flask and XGBoost`.
5. Choose Public or Private according to your college/project requirements.
6. Do not initialize another README if you are going to push this prepared folder.
7. Select **Create repository**.

## C. Push using Git (recommended)
Open the terminal in the extracted project directory and run:

```bash
git init
git add .
git status
git commit -m "Add complete Rising Water flood prediction project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/rising-water-flood-prediction.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username and use the exact repository URL shown by GitHub. GitHub may ask you to authenticate through a browser or credential manager.

## D. Upload through the GitHub website (alternative)
1. Open the newly created repository.
2. Select **Add file → Upload files**.
3. Extract the ZIP first; upload the files and folders inside the extracted project, not the ZIP itself.
4. Wait for all uploads to finish.
5. Enter a commit message such as `Add complete project files`.
6. Select **Commit changes**.

## E. Verify the repository
After pushing, open the GitHub repository and check:
- README renders on the front page.
- `app.py` and `train_model.py` are present.
- `dataset/flood.csv` is present.
- The two model files are present in `models/`.
- All four templates and the CSS/JavaScript/image assets are present.
- No `.venv`, cache folders, passwords, API keys or personal files were uploaded.

## F. Run after cloning
```bash
git clone https://github.com/YOUR-USERNAME/rising-water-flood-prediction.git
cd rising-water-flood-prediction
python -m venv .venv
# Activate the environment using the command for your operating system.
pip install -r requirements.txt
python app.py
```

## G. Common Git errors
- `remote origin already exists`: run `git remote set-url origin YOUR-REPOSITORY-URL`.
- `src refspec main does not match any`: make the first commit, then push.
- `rejected (fetch first)`: the remote has commits not in your local branch. Fetch and reconcile before pushing; do not force-push blindly.
- Large file warning: the included dataset is about 2.3 MB and the saved model is under 1 MB, so this prepared project should be within normal GitHub file limits.
