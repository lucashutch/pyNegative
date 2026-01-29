---
description: create a feature branch and open a pull request
---

This workflow automates the process of creating a new branch and opening a PR.

1. Create a new branch
```bash
git checkout -b <branch-name>
```

2. Make changes and commit
```bash
git add .
git commit -m "<message>"
```

// turbo
3. Push the branch to origin
```bash
git push origin <branch-name>
```

// turbo
4. Create the pull request via GitHub CLI
```bash
gh pr create --title "<title>" --body "<body>"
```
