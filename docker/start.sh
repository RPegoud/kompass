#!/usr/bin/env bash
# Pod entrypoint: SSH access, repo checkout/install, Nsight preflight, then idle.
#
# Env:
#   PUBLIC_KEY        authorized SSH key (Runpod injects this from your account settings)
#   KOMPASS_DIR       repo location, default /workspace/kompass (keep it on the volume)
#   KOMPASS_REPO_URL  optional; cloned into KOMPASS_DIR if it doesn't exist yet
#                     (private repo: https://<token>@github.com/<user>/kompass.git)
set -uo pipefail

KOMPASS_DIR="${KOMPASS_DIR:-/workspace/kompass}"

# 1. Make the container env visible to SSH sessions (see Dockerfile).
export -p | grep -vE '^declare -x (HOME|PWD|OLDPWD|SHLVL|HOSTNAME|TERM|_)=' > /etc/kompass-env.sh
chmod 600 /etc/kompass-env.sh

# 2. SSH (needed for VS Code Remote-SSH, rsync, scp).
if [[ -n "${PUBLIC_KEY:-}" ]]; then
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    grep -qxF "$PUBLIC_KEY" /root/.ssh/authorized_keys 2>/dev/null \
        || echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    ssh-keygen -A
    /usr/sbin/sshd && echo "[kompass] sshd started"
else
    echo "[kompass] PUBLIC_KEY not set, skipping sshd"
fi

# 3. Repo: clone if asked, then install the project (editable) into /opt/venv.
if [[ -n "${KOMPASS_REPO_URL:-}" && ! -d "$KOMPASS_DIR/.git" ]]; then
    git clone "$KOMPASS_REPO_URL" "$KOMPASS_DIR"
fi
if [[ -f "$KOMPASS_DIR/pyproject.toml" ]]; then
    (cd "$KOMPASS_DIR" && uv sync --frozen) && echo "[kompass] installed from $KOMPASS_DIR"
else
    echo "[kompass] no repo at $KOMPASS_DIR; rsync/clone it there and run 'uv sync --frozen'"
fi

# 4. Tell us right away whether `kompass profile` can work on this host.
ncu-check

exec sleep infinity
