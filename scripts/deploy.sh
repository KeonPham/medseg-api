#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────
# MedSegAPI — One-click deploy script for Oracle Cloud ARM VM
# Target: Ubuntu 22.04+ on Ampere A1 (aarch64)
#
# Usage:
#   1. SSH into your Oracle Cloud VM
#   2. Clone the repo:  git clone <repo-url> medseg-api && cd medseg-api
#   3. Copy model weights into models/ (see below)
#   4. Run:  bash scripts/deploy.sh your-domain.com
# ───────────────────────────────────────────────────────────────
set -euo pipefail

DOMAIN="${1:-}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ─── Colors ───
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ─── Pre-checks ───
info "MedSegAPI deployment starting..."
echo ""

if [ -z "$DOMAIN" ]; then
    echo "Usage: bash scripts/deploy.sh <your-domain.com>"
    echo ""
    echo "  Pass your custom domain (DNS must already point to this server's IP)."
    echo "  For testing without a domain, use:  bash scripts/deploy.sh localhost"
    exit 1
fi

# Check model weights exist
for model_dir in hybrid cnn_only vit_only; do
    if ! ls "$REPO_DIR/models/$model_dir/"*.pth >/dev/null 2>&1; then
        fail "Missing model weights in models/$model_dir/. Copy .pth files first."
    fi
done
ok "Model weights found"

# ─── Step 1: Install Docker ───
if command -v docker &>/dev/null; then
    ok "Docker already installed: $(docker --version)"
else
    info "Installing Docker..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        sudo tee /etc/apt/sources-list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo usermod -aG docker "$USER"
    ok "Docker installed"
fi

# ─── Step 2: Open firewall (Oracle Cloud uses iptables) ───
info "Configuring firewall..."
# Oracle Cloud Ubuntu images use iptables by default
sudo iptables -I INPUT -p tcp --dport 80  -j ACCEPT 2>/dev/null || true
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null || true
# Persist rules
if command -v netfilter-persistent &>/dev/null; then
    sudo netfilter-persistent save 2>/dev/null || true
else
    sudo apt-get install -y -qq iptables-persistent 2>/dev/null || true
    sudo netfilter-persistent save 2>/dev/null || true
fi
ok "Ports 80 and 443 open"

# ─── Step 3: Create .env ───
cd "$REPO_DIR"
if [ ! -f .env ]; then
    info "Creating .env file..."
    PG_PASS=$(openssl rand -hex 16)
    cat > .env <<EOF
POSTGRES_PASSWORD=$PG_PASS
DOMAIN=$DOMAIN
EOF
    ok ".env created (Postgres password auto-generated)"
else
    # Update domain in existing .env
    if grep -q "^DOMAIN=" .env; then
        sed -i "s/^DOMAIN=.*/DOMAIN=$DOMAIN/" .env
    else
        echo "DOMAIN=$DOMAIN" >> .env
    fi
    ok ".env already exists, domain updated to $DOMAIN"
fi

# ─── Step 4: Build and start ───
info "Building Docker image (this may take 5-10 minutes on first run)..."
sudo docker compose -f docker-compose.prod.yml build

info "Starting services..."
sudo docker compose -f docker-compose.prod.yml up -d

# Wait for health check
info "Waiting for API to become healthy..."
for i in $(seq 1 30); do
    if sudo docker compose -f docker-compose.prod.yml exec -T api curl -sf http://localhost:8000/api/v1/health >/dev/null 2>&1; then
        ok "API is healthy!"
        break
    fi
    if [ "$i" -eq 30 ]; then
        fail "API did not become healthy in 60s. Check logs: docker compose -f docker-compose.prod.yml logs api"
    fi
    sleep 2
done

# ─── Step 5: Create API key ───
info "Creating API key..."
API_KEY_OUTPUT=$(sudo docker compose -f docker-compose.prod.yml exec -T api python -m scripts.create_api_key --name production 2>&1 || true)
echo "$API_KEY_OUTPUT"

# ─── Done ───
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
if [ "$DOMAIN" = "localhost" ]; then
    ok "MedSegAPI is running at http://$(curl -s ifconfig.me):80"
else
    ok "MedSegAPI is running at https://$DOMAIN"
fi
echo ""
echo "  Useful commands:"
echo "    Logs:     sudo docker compose -f docker-compose.prod.yml logs -f"
echo "    Stop:     sudo docker compose -f docker-compose.prod.yml down"
echo "    Restart:  sudo docker compose -f docker-compose.prod.yml restart"
echo "    API key:  sudo docker compose -f docker-compose.prod.yml exec api python -m scripts.create_api_key"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
