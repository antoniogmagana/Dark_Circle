#!/usr/bin/env bash
# Bootstrap a Linux host for building inference-engine images and
# running the test suite. Designed for "duplicate from a repo clone"
# customer setups.
#
# What this checks for and (with permission) installs:
#   - docker, kubectl, helm, kind on PATH
#   - poetry venv with the test dep group in inference-engine/
#   - inference-protos installed into the poetry venv
#   - CPU-only torch + torchaudio in the poetry venv
#   - `poetry run pytest tests/ -v` passes as a self-check
#
# What this does NOT do:
#   - Touch crl-train. Customers should select a pre-built bundle from
#     crl-bundles/ via the CRL_BUNDLE env var (see crl-bundles/README.md).
#   - Configure kubectl context, GPU drivers, or registry auth.
#
# Idempotent: re-running with everything already installed exits clean
# without prompts.
#
# Platform: Ubuntu / Debian only. Other distros, follow the manual
# steps in the inference-engine README's "Build and Deploy" section.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- 0. Platform check ------------------------------------------------------
if [ ! -f /etc/os-release ]; then
    echo "ERROR: cannot detect OS; this script supports Ubuntu / Debian only." >&2
    exit 1
fi
. /etc/os-release
case "$ID" in
    ubuntu|debian) ;;
    *)
        echo "ERROR: $PRETTY_NAME is not supported by this script." >&2
        echo "       Follow the manual steps in inference-engine/README.md" >&2
        echo "       under 'Build and Deploy' for other platforms." >&2
        exit 1
        ;;
esac

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    if ! command -v sudo >/dev/null 2>&1; then
        echo "ERROR: this script needs root or sudo. Re-run as root or install sudo." >&2
        exit 1
    fi
    SUDO=sudo
fi

# ---- Helpers ---------------------------------------------------------------
prompt_install() {
    # prompt_install <tool-name> <description>
    local tool="$1"
    local desc="$2"
    if [ "${ASSUME_YES:-0}" = "1" ]; then
        return 0
    fi
    printf "Install %s (%s)? [y/N] " "$tool" "$desc"
    read -r ans
    case "$ans" in
        y|Y|yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

# ---- 1. Tooling on PATH ----------------------------------------------------
echo "=== checking build-host tooling ==="

if command -v docker >/dev/null 2>&1; then
    echo "  ✓ docker: $(command -v docker)"
else
    echo "  ✗ docker: missing"
    if prompt_install docker "container runtime, official Docker apt repo"; then
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq ca-certificates curl gnupg
        $SUDO install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/${ID}/gpg \
            | $SUDO gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        $SUDO chmod a+r /etc/apt/keyrings/docker.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${ID} $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
            | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq docker-ce docker-ce-cli containerd.io
        # User must log out and back in for group membership to take
        # effect — we don't try to fix that automatically.
        $SUDO usermod -aG docker "$USER" || true
        echo "  installed docker. You may need to log out and back in for group changes."
    else
        echo "  skipping docker install — build_containers.sh will fail without it." >&2
    fi
fi

if command -v kubectl >/dev/null 2>&1; then
    echo "  ✓ kubectl: $(command -v kubectl)"
else
    echo "  ✗ kubectl: missing"
    if prompt_install kubectl "Kubernetes CLI, official Google apt repo"; then
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq apt-transport-https ca-certificates curl gnupg
        $SUDO install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.31/deb/Release.key \
            | $SUDO gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
        $SUDO chmod a+r /etc/apt/keyrings/kubernetes-apt-keyring.gpg
        echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.31/deb/ /" \
            | $SUDO tee /etc/apt/sources.list.d/kubernetes.list >/dev/null
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq kubectl
    else
        echo "  skipping kubectl install — deployment commands will fail without it." >&2
    fi
fi

if command -v helm >/dev/null 2>&1; then
    echo "  ✓ helm: $(command -v helm)"
else
    echo "  ✗ helm: missing"
    if prompt_install helm "Helm 3 chart installer, official one-liner"; then
        curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | $SUDO bash
    else
        echo "  skipping helm install — chart installs will fail without it." >&2
    fi
fi

if command -v kind >/dev/null 2>&1; then
    echo "  ✓ kind: $(command -v kind)"
else
    echo "  ✗ kind: missing"
    if prompt_install kind "Kubernetes-in-Docker, single-binary install"; then
        local_arch="$(dpkg --print-architecture)"
        case "$local_arch" in
            amd64) kind_arch=amd64 ;;
            arm64) kind_arch=arm64 ;;
            *) echo "    unsupported arch $local_arch — install kind manually." >&2 ; kind_arch="" ;;
        esac
        if [ -n "$kind_arch" ]; then
            curl -fsSL "https://kind.sigs.k8s.io/dl/latest/kind-linux-${kind_arch}" \
                -o /tmp/kind
            chmod +x /tmp/kind
            $SUDO mv /tmp/kind /usr/local/bin/kind
        fi
    else
        echo "  skipping kind install — local_smoke.sh will fail without it." >&2
    fi
fi

# ---- 2. Poetry venv with test deps -----------------------------------------
echo "=== installing inference-engine python deps ==="

if ! command -v poetry >/dev/null 2>&1; then
    echo "  ✗ poetry: missing"
    if prompt_install poetry "Python dependency manager, official installer"; then
        curl -sSL https://install.python-poetry.org | python3 -
        # poetry's installer puts it under ~/.local/bin; tell the user
        # to add that to PATH for this shell.
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "  skipping poetry install — tests will not run without it." >&2
        exit 1
    fi
fi

cd "$ROOT"
echo "=== poetry install --with test ==="
poetry install --with test

echo "=== installing inference-protos into poetry venv ==="
poetry run pip install -q ./inference-protos

# torch / torchaudio: install CPU-only wheels. The default PyPI wheels
# pull in CUDA libs that fail to dlopen on cudaless hosts (the same
# pattern as src/ingestor/Dockerfile:18-24).
echo "=== installing CPU-only torch + torchaudio ==="
poetry run pip install -q \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchaudio

# ---- 3. Self-check: pytest -------------------------------------------------
echo "=== running test suite ==="
if poetry run pytest tests/ -q; then
    echo
    echo "=== install_build_host.sh complete ==="
    echo
    echo "Next steps:"
    echo "  1. Build images:    bash scripts/build_containers.sh"
    echo "  2. Local smoke:     bash scripts/local_smoke.sh"
    echo "  3. View results:    bash scripts/tail_egress.sh  (after cluster is up)"
else
    echo
    echo "=== test suite failed ===" >&2
    echo "Inspect the output above. The image build will likely also fail." >&2
    exit 1
fi
