#!/bin/bash
set -e
env >> /etc/environment
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq git curl wget nano vim rsync unzip jq htop ncdu sqlite3 pciutils lshw tmux

if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

if ! pgrep -f "ollama serve" >/dev/null; then
  mkdir -p /var/log/ollama
  tmux kill-session -t ollama 2>/dev/null || true
  tmux new-session -d -s ollama \
    "OLLAMA_MODELS=/data/models OLLAMA_HOST=0.0.0.0:11434 ollama serve 2>&1 | tee -a /var/log/ollama/server.log"
fi

grep -q OLLAMA_MODELS /root/.bashrc || echo 'export OLLAMA_MODELS=/data/models' >> /root/.bashrc
grep -q OLLAMA_HOST /root/.bashrc   || echo 'export OLLAMA_HOST=0.0.0.0:11434'   >> /root/.bashrc

if [ -d /data/leo_trident ] && [ ! -x /data/leo_trident/.venv/bin/python ]; then
  cd /data/leo_trident
  rm -rf .venv leo_trident.egg-info
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip wheel setuptools
  .venv/bin/pip install -q -r requirements.txt
  .venv/bin/pip install -q -e .
fi

git config --global user.email "${GITHUB_EMAIL:-bwturley1@gmail.com}"
git config --global user.name  "${GITHUB_USER:-BWTurley}"
date > /data/.provisioned_at

# Auto-start leo-trident health service
if [ -d /data/leo_trident ] && ! tmux has-session -t leo-health 2>/dev/null; then
  tmux new-session -d -s leo-health \
    "cd /data/leo_trident && source .venv/bin/activate && python scripts/run_health.py 2>&1 | tee -a /var/log/leo_health.log"
fi

# Ensure cron is running (for backup scheduling)
if ! pgrep -af cron >/dev/null; then
  service cron start 2>/dev/null || /usr/sbin/cron
fi

# Restore git credentials from persistent volume
if [ -f /data/.creds/.git-credentials ]; then
  cp /data/.creds/.git-credentials /root/.git-credentials
  chmod 600 /root/.git-credentials
  git config --global credential.helper 'store --file /root/.git-credentials'
fi
