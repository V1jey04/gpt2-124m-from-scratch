#!/bin/sh
# Publish the code to GitHub, the weights to a Hugging Face model repo, and the demo to a Space.
# Everything it creates is PUBLIC. Review the README, model card and Space files first.
#
#   hf auth login          # once, with a write token from huggingface.co/settings/tokens
#   sh scripts/publish.sh  # run from the project root
#
# Re-running is safe: repos that exist are reused and files are updated.
set -eu

NAME="${NAME:-gpt2-124m-from-scratch}"
GH_USER="$(gh api user --jq .login)"
HF_USER="$(hf auth whoami 2>/dev/null | sed -n 's/^user[=:] *//p; s/^username[=:] *//p' | head -1)"
[ -n "$HF_USER" ] || { echo "Not logged in to Hugging Face: run 'hf auth login' first."; exit 1; }
echo "GitHub: $GH_USER/$NAME   Hugging Face: $HF_USER/$NAME (model and Space)"

# fill the username placeholders in the README, model card and Space files
for f in README.md hf_release/model/README.md hf_release/space/README.md hf_release/space/app.py; do
  sed -i.bak "s/HF_USER/$HF_USER/g; s/GITHUB_USER/$GH_USER/g" "$f" && rm -f "$f.bak"
done

# 1. weights (model.safetensors + config.json + tokenizer.json + model card)
[ -f hf_release/model/model.safetensors ] || .venv/bin/python scripts/export_hf.py
hf repos create "$HF_USER/$NAME" --type model --public --exist-ok
hf upload "$HF_USER/$NAME" hf_release/model . --repo-type model --commit-message "Upload weights and model card"

# 2. demo Space (free CPU)
hf repos create "$HF_USER/$NAME" --type space --space-sdk gradio --public --exist-ok
hf upload "$HF_USER/$NAME" hf_release/space . --repo-type space --commit-message "Chat demo"

# 3. code (commit identity for this repo only: GitHub name + private no-reply email)
if ! git config user.email >/dev/null; then
  GH_ID="$(gh api user --jq .id)"
  git config user.name "$(gh api user --jq '.name // .login')"
  git config user.email "$GH_ID+$GH_USER@users.noreply.github.com"
fi
git add -A
git commit -q -F - <<'MSG' || true
Publish: README with results, Hugging Face export and Space demo

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
if ! gh repo view "$GH_USER/$NAME" >/dev/null 2>&1; then
  gh repo create "$GH_USER/$NAME" --public --source . --remote origin \
    --description "124M GPT-2-size chat model: tokenizer, transformer and training written from scratch, trained on free Kaggle GPUs"
fi
git push -u origin main

echo
echo "Code:  https://github.com/$GH_USER/$NAME"
echo "Model: https://huggingface.co/$HF_USER/$NAME"
echo "Demo:  https://huggingface.co/spaces/$HF_USER/$NAME   (takes a few minutes to build)"
