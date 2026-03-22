#!/bin/bash
# ================================================================
# Wan2GP + Wan2GP-API - New Server Setup Script
# ================================================================
# Is script ko ek baar chalao naye server pe.
# Yeh script GPU container (wan2gp) aur API stack (wan2gp-api)
# dono ko ek saath setup karta hai.
#
# Usage:
#   chmod +x setup_new_server.sh
#   sudo ./setup_new_server.sh
# ================================================================

set -e  # koi bhi command fail ho to script rok do

# ── CONFIG — Apne server ke hisaab se change karo ──────────────
DISK_ROOT="/nvme0n1-disk"                      # Main disk path
WAN2GP_CODE_DIR="$DISK_ROOT/nvme01/Wan2GP"    # wan2gp Python code (cloned repo)
WAN2GP_API_DIR="$DISK_ROOT/nvme01/wan2gp-api" # wan2gp-api code
DATA_DIR="$DISK_ROOT/wan2gp_data"             # Runtime data (outputs, settings)

# Kitne GPUs hain? (1 ya 3)
GPU_COUNT=3   # 1-GPU server ke liye isko 1 kar lo

# Local Docker image name (locally built from Wan2GP source)
WAN2GP_IMAGE="wan2gp-local:latest"

# RTX A5000 = sm_8.6, A100 = sm_8.0, RTX 4090 = sm_8.9
CUDA_ARCHITECTURES="8.6"

# ── Colors ──────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
info() { echo -e "${BLUE}[ℹ]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ================================================================
echo ""
echo "================================================================"
echo "   Wan2GP + Wan2GP-API - Server Setup"
echo "================================================================"
echo ""

# ── STEP 1: System Requirements Check ───────────────────────────
info "Step 1: System requirements check kar rahe hain..."

command -v docker &>/dev/null || err "Docker install nahi hai! Pehle Docker install karo."
command -v git    &>/dev/null || err "Git install nahi hai!"

# Docker nvidia runtime check
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    warn "NVIDIA Docker runtime nahi mila. GPU support ke liye zaruri hai."
    warn "Chalao: sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker"
fi

# Check CUDA GPUs
GPU_LIST=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "")
if [ -z "$GPU_LIST" ]; then
    warn "NVIDIA GPU nahi mila ya nvidia-smi nahi hai!"
else
    log "GPUs found:"
    echo "$GPU_LIST" | nl -ba -v0
fi

log "Requirements check complete."

# ── STEP 2: Directories banao ───────────────────────────────────
info "Step 2: Data directories bana rahe hain..."

mkdir -p "$DATA_DIR/outputs"
mkdir -p "$DATA_DIR/uploads"

for i in $(seq 0 $((GPU_COUNT - 1))); do
    mkdir -p "$DATA_DIR/gpu${i}/settings"
    log "  Directory bana: $DATA_DIR/gpu${i}/settings"
done

# Permissions
chmod -R 777 "$DATA_DIR"
log "Data directories ready: $DATA_DIR"

# ── STEP 3: Code directories check karo ─────────────────────────
info "Step 3: Code repositories check kar rahe hain..."

# Wan2GP (GPU container ka Python code)
if [ ! -f "$WAN2GP_CODE_DIR/wgp.py" ]; then
    warn "Wan2GP code nahi mila ya wgp.py missing: $WAN2GP_CODE_DIR"
    info "Clone kar rahe hain..."
    git clone https://github.com/deepbeepmeep/Wan2GP.git "$WAN2GP_CODE_DIR"
    log "Wan2GP cloned: $WAN2GP_CODE_DIR"
else
    log "Wan2GP code found: $WAN2GP_CODE_DIR (wgp.py present)"
fi

# Wan2GP-API (REST API layer)
if [ ! -d "$WAN2GP_API_DIR/.git" ]; then
    err "wan2gp-api directory missing: $WAN2GP_API_DIR\nPehle clone karo:\n  git clone https://github.com/surajit20072003/wan2gp-api.git $WAN2GP_API_DIR"
else
    log "wan2gp-api code found: $WAN2GP_API_DIR"
fi

# ── STEP 4: Docker Network banao ────────────────────────────────
info "Step 4: Docker network setup kar rahe hain..."

if docker network ls | grep -q "wan2gp_network"; then
    warn "wan2gp_network pehle se exist karta hai, skip kar rahe hain."
else
    docker network create wan2gp_network
    log "Docker network 'wan2gp_network' bana diya."
fi

# ── STEP 5: Default settings files banao ────────────────────────
info "Step 5: Per-GPU default settings files bana rahe hain..."

for i in $(seq 0 $((GPU_COUNT - 1))); do
    SETTINGS_FILE="$DATA_DIR/gpu${i}/settings/default_settings.json"
    if [ ! -f "$SETTINGS_FILE" ]; then
        cat > "$SETTINGS_FILE" <<EOF
{
  "model": "ltx-2-19b-distilled",
  "video_length": 81,
  "resolution": "1280x720",
  "steps": 8,
  "seed": -1,
  "loras": {}
}
EOF
        log "Settings file bana: $SETTINGS_FILE"
    else
        warn "Settings file pehle se hai: $SETTINGS_FILE"
    fi
done

# ── STEP 6: Docker Image Build karo ─────────────────────────────
info "Step 6: Wan2GP Docker image build kar rahe hain..."
info "  ⚠️  Pehli baar ~20-30 min lagega (SageAttention compile hoti hai)"
info "  CUDA Architecture: ${CUDA_ARCHITECTURES} (RTX A5000 ke liye)"

# Check karo ki image pehle se bani hai
if docker image inspect "$WAN2GP_IMAGE" &>/dev/null; then
    warn "Image '$WAN2GP_IMAGE' pehle se exist karti hai."
    read -p "    Dobara build karna hai? (y/N): " REBUILD
    if [[ "$REBUILD" =~ ^[Yy]$ ]]; then
        DOCKER_BUILDKIT=1 docker build \
            --build-arg CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
            -t "$WAN2GP_IMAGE" \
            "$WAN2GP_CODE_DIR"
        log "Image rebuild complete: $WAN2GP_IMAGE"
    else
        log "Existing image use karenge: $WAN2GP_IMAGE"
    fi
else
    DOCKER_BUILDKIT=1 docker build \
        --build-arg CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
        -t "$WAN2GP_IMAGE" \
        "$WAN2GP_CODE_DIR"
    log "Image build complete: $WAN2GP_IMAGE"
fi

# ── STEP 7: GPU Containers start karo ───────────────────────────
info "Step 7: Wan2GP GPU containers start kar rahe hain..."

for i in $(seq 0 $((GPU_COUNT - 1))); do
    CONTAINER_NAME="wan2gp-gpu${i}"

    # Pehle se chal raha hai to hatao
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        warn "Container ${CONTAINER_NAME} pehle se hai, hat rahe hain..."
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    fi

    info "GPU ${i}: Container shuru kar rahe hain (${CONTAINER_NAME})..."

    docker run -d \
        --name "$CONTAINER_NAME" \
        --runtime=nvidia \
        --gpus "\"device=${i}\"" \
        --restart=always \
        --network=wan2gp_network \
        --shm-size=8g \
        -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
        -v "${WAN2GP_CODE_DIR}:/workspace" \
        -v "${DATA_DIR}/outputs:/workspace/outputs" \
        -v "${DATA_DIR}/gpu${i}/settings:/workspace/settings" \
        -v "${DATA_DIR}/uploads:/nvme0n1-disk/wan2gp_data/uploads" \
        --entrypoint /bin/bash \
        "$WAN2GP_IMAGE" \
        -c "tail -f /dev/null"

    log "  GPU ${i} container shuru hua: ${CONTAINER_NAME}"
done

# ── STEP 8: API Stack start karo ────────────────────────────────
info "Step 8: Wan2GP API stack start kar rahe hain (FastAPI + Redis)..."

cd "$WAN2GP_API_DIR"

if [ $GPU_COUNT -eq 1 ]; then
    COMPOSE_FILE="docker-compose.yml"
else
    COMPOSE_FILE="docker-compose-3gpu.yml"
fi

docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true
DOCKER_BUILDKIT=1 docker compose -f "$COMPOSE_FILE" up -d --build

log "API stack containers start ho gaye."

# ── STEP 9: Health Check ─────────────────────────────────────────
info "Step 9: Health check kar rahe hain (30 second wait)..."
sleep 30

# API health check
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "X-API-Key: default-secret-key" http://localhost:8000/health 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    log "API healthy hai! http://localhost:8000"
else
    warn "API abhi ready nahi hai (HTTP: $HTTP_CODE). Thodi der mein try karo:"
    warn "    Bahar se port 8000 pe access test karein:"
    warn "  curl -H \"X-API-Key: default-secret-key\" http://localhost:8000/health"
fi

# GPU container status check
for i in $(seq 0 $((GPU_COUNT - 1))); do
    CONTAINER="wan2gp-gpu${i}"
    STATUS=$(docker inspect --format='{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo "not found")
    if [ "$STATUS" = "running" ]; then
        log "  GPU ${i} container: running ✓"
    else
        warn "  GPU ${i} container status: $STATUS"
    fi
done

# ── STEP 10: Firewall ports open karo ───────────────────────────
info "Step 10: Firewall ports open kar rahe hain..."

if command -v ufw &>/dev/null; then
    ufw allow 8000/tcp 2>/dev/null && log "Port 8000 open kar diya (ufw)." || warn "ufw port open nahi hua."
else
    warn "ufw nahi mila. Manually port 8000 open karo agar chahiye."
fi

# ── FINAL SUMMARY ────────────────────────────────────────────────
echo ""
echo "================================================================"
echo -e "${GREEN}   Setup Complete!${NC}"
echo "================================================================"
echo ""
echo "  API Server:    http://localhost:8000"
echo "  Dashboard:     http://localhost:8000/dashboard"
echo "  Swagger Docs:  http://localhost:8000/docs"
echo "  Admin Panel:   http://localhost:8000/admin"
echo ""
echo "  GPU Containers:"
for i in $(seq 0 $((GPU_COUNT - 1))); do
    echo "    wan2gp-gpu${i}  ->  GPU ${i}"
done
echo ""
echo "  Data Directory: $DATA_DIR"
echo "    ├── outputs/         (generated videos)"
echo "    ├── uploads/         (input images/audio)"
for i in $(seq 0 $((GPU_COUNT - 1))); do
    echo "    ├── gpu${i}/settings/ (GPU ${i} job settings)"
done
echo ""
echo "  Useful Commands:"
echo "    docker ps                          # sab containers dekho"
echo "    docker logs wan2gp-api -f          # API logs dekho"
echo "    docker logs wan2gp-gpu0 -f         # GPU 0 logs dekho"
echo "    curl -H \"X-API-Key: default-secret-key\" http://localhost:8000/health  # API health check"
echo "    curl -H \"X-API-Key: default-secret-key\" http://localhost:8000/queue   # Queue status"
echo ""
echo "  ℹ️  Note: Pehli video generation mein models download honge"
echo "     (~20-40GB), uske baad fast rahega."
echo ""
echo "================================================================"
