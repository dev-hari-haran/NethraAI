# ==============================================================================
# NethraAI - 1-Click Hugging Face Space Deployer
# ==============================================================================
param(
    [Parameter(Mandatory=$false)]
    [string]$SpaceUrl = ""
)

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   NethraAI Backend -> Hugging Face Spaces Deployer       " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

if (-not $SpaceUrl) {
    Write-Host "Enter your Hugging Face Space Git URL:" -ForegroundColor Yellow
    Write-Host "Example: https://huggingface.co/spaces/YOUR_USERNAME/nethra-ai-backend" -ForegroundColor Gray
    $SpaceUrl = Read-Host "Space Git URL"
}

if (-not $SpaceUrl) {
    Write-Error "Space URL is required. Exiting."
    exit 1
}

# 1. Verify weights
if (-not (Test-Path "checkpoints/best_model.pth")) {
    Write-Error "checkpoints/best_model.pth not found! Model file is required."
    exit 1
}

Write-Host "`n[1/5] Initializing Git LFS..." -ForegroundColor Green
git lfs install

Write-Host "[2/5] Setting up Git LFS tracking for checkpoints..." -ForegroundColor Green
git lfs track "checkpoints/*.pth"
git add .gitattributes

Write-Host "[3/5] Creating temporary deployment snapshot..." -ForegroundColor Green
git branch -D hf-deploy 2>$null
git checkout -b hf-deploy

# Add core backend files and force-add weights to this deployment branch
git add Dockerfile requirements-backend.txt .dockerignore
git add config.py model.py infer.py shap_report.py preprocess.py preprocess_idrid.py
git add vessel_model.py vessel_dataset.py vessel_segmentor.py
git add webapp/
git add -f checkpoints/best_model.pth checkpoints/vessel_unet.pth 2>$null

git commit -m "Deploy NethraAI backend & models to Hugging Face Spaces" --allow-empty

Write-Host "[4/5] Connecting to Hugging Face Space remote..." -ForegroundColor Green
git remote remove space 2>$null
git remote add space $SpaceUrl

Write-Host "[5/5] Pushing to Hugging Face Space (uploading code & model weights)..." -ForegroundColor Green
Write-Host "Note: When prompted for credentials, use your Hugging Face Username and an Access Token (with Write permission) as password." -ForegroundColor Yellow

git push -u space hf-deploy:main --force

# Return to main branch
git checkout main
git branch -D hf-deploy 2>$null

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " Deployment pushed to Hugging Face Spaces!" -ForegroundColor Green
Write-Host " Space URL: $SpaceUrl" -ForegroundColor Cyan
Write-Host " Your API will be live at: https://<username>-<space-name>.hf.space" -ForegroundColor Cyan
Write-Host "==========================================================`n" -ForegroundColor Green
