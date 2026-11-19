# Publish RAAI-SGC 2.0 to GitHub

This guide **does not publish your laptop files automatically**. Run it in the Ubuntu terminal inside the existing project directory.

## 1. Check Git and GitHub CLI

```bash
cd ~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT_READY
git --version
gh --version
```

If `gh` is missing: `sudo apt update && sudo apt install gh`.

Log into the **intended GitHub account**:

```bash
gh auth login
gh auth status
```

## 2. Add README and .gitignore

Copy the supplied `README.md` and `.gitignore` into the root of your existing project. Keep a backup of any existing `.gitignore` if it contains useful ignore rules.

## 3. Verify what would be published BEFORE committing

```bash
git init

git status --short

git add README.md .gitignore
# Add project directories only after checking private/large data rules:
git add app src dataset_engineering training evaluation tests docs notebooks 2>/dev/null || true
# Add other actual project files selectively; never blindly add secrets or adapter files.
git status --short
```

If you intend to include `data/processed/master.csv` publicly, first verify redistribution terms and attribution. Keep the repo private until this is done.

Inspect tracked paths before the first commit:

```bash
git diff --cached --name-only
```

Look for `.env`, passwords, API tokens, names/review decisions, trained weights, or files you lack permission to redistribute. Unstage anything questionable with `git restore --staged PATH` (or `git reset HEAD PATH` if necessary).

## 4. Commit and create a private repository

```bash
git add README.md .gitignore
git commit -m "docs: publish RAAI-SGC 2.0 industry capstone"
git branch -M main

gh repo create RAAI-SGC-2.0 --private --source=. --remote=origin --push
```

`gh repo create` may ask for confirmation. If the repository already exists or `origin` is already set, do not force-push: inspect `git remote -v`, then push to the correct remote with `git push -u origin main` once verified.

## 5. Optional: publish the rest of the intended source tree

When the contents are reviewed:

```bash
git add app src dataset_engineering training evaluation tests docs
# Add only existing folders; adjust as needed.
git status --short
git diff --cached --name-only
git commit -m "feat: add RAAI-SGC application and training pipeline"
git push
```

## 6. Verify

```bash
gh repo view --web

git status
```

If the project has never been initialized as a Git repository, do **not** run `git push --force`. Preserve your existing work and history.

## Note about clone reproducibility

The recommended `.gitignore` excludes virtual environments, local LoRA adapters, private approval data, and generated experiment outputs. A fresh clone requires dependency setup plus access to appropriately licensed source data and any adapter you choose to demonstrate.
