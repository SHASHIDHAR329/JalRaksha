$ErrorActionPreference = "Stop"

$Root = "C:\JalRaksha"
$BackupRoot = "C:\JalRaksha_Backups"

$Stamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$Checkpoint = Join-Path $BackupRoot "JalRaksha_Checkpoint_$Stamp"

New-Item -ItemType Directory -Path $Checkpoint -Force | Out-Null
New-Item -ItemType Directory -Path "$Checkpoint\environment" -Force | Out-Null
New-Item -ItemType Directory -Path "$Checkpoint\vscode" -Force | Out-Null
New-Item -ItemType Directory -Path "$Checkpoint\external_sources" -Force | Out-Null

Write-Host ""
Write-Host "============================================================"
Write-Host "JALRAKSHA CHECKPOINT BACKUP"
Write-Host "============================================================"

# ------------------------------------------------------------
# 1. Project inventory
# ------------------------------------------------------------

Get-ChildItem $Root -Recurse -File |
    Select-Object FullName, Length, LastWriteTime |
    Sort-Object FullName |
    Out-File "$Checkpoint\PROJECT_FILE_INVENTORY.txt"

# ------------------------------------------------------------
# 2. Conda / Python environment
# ------------------------------------------------------------

conda env export -n jalraksha |
    Out-File "$Checkpoint\environment\jalraksha_environment.yml" -Encoding utf8

conda list -n jalraksha |
    Out-File "$Checkpoint\environment\jalraksha_conda_list.txt" -Encoding utf8

python --version |
    Out-File "$Checkpoint\environment\python_version.txt" -Encoding utf8

python -m pip freeze |
    Out-File "$Checkpoint\environment\pip_freeze.txt" -Encoding utf8

# ------------------------------------------------------------
# 3. VS Code extensions
# ------------------------------------------------------------

$CodeCommand = Get-Command code -ErrorAction SilentlyContinue

if ($CodeCommand) {
    code --list-extensions --show-versions |
        Out-File "$Checkpoint\vscode\vscode_extensions.txt" -Encoding utf8
}
else {
    "VS Code CLI command 'code' was not available." |
        Out-File "$Checkpoint\vscode\vscode_extensions.txt" -Encoding utf8
}

# Copy VS Code project settings if present
if (Test-Path "$Root\.vscode") {
    Copy-Item "$Root\.vscode" "$Checkpoint\vscode\.vscode" -Recurse -Force
}

# ------------------------------------------------------------
# 4. Current project state
# ------------------------------------------------------------

@"
JALRAKSHA PROJECT CHECKPOINT
Created: $(Get-Date)

PROJECT ROOT
-----------
$Root

PYTHON ENVIRONMENT
------------------
Environment: jalraksha
Python: 3.11.x
GPU: NVIDIA RTX 3050 Laptop GPU 6 GB
PyTorch: CUDA-enabled
NeuralOperator: installed

COMPLETED PIPELINE
------------------
Dataset
  -> DualSPHysics
  -> FlowTool
  -> clean hydrograph
  -> D-Flow FM
  -> flood-depth extraction
  -> regular-grid conversion
  -> FNO-ready dataset

CURRENT DATASET
---------------
10 scenarios

FNO DATASET
-----------
Directory:
C:\JalRaksha\outputs\fno_dataset

X shape:
(10, 7, 16, 128)

Y shape:
(10, 1, 16, 128)

Mask:
(16, 128)

Validation:
X NaNs = 0
Y NaNs = 0

Valid grid cells:
1744 / 2048

Maximum depth range:
0.206236 m to 0.231411 m

CURRENT MANIFEST
----------------
C:\JalRaksha\outputs\dataset_manifest.csv

SCENARIOS
---------
scenario_01
scenario_02
scenario_03
scenario_04
scenario_05
scenario_06
scenario_07
scenario_08
scenario_09
scenario_10

NEXT DEVELOPMENT TASK
---------------------
Train and validate the first FNO prototype.

IMPORTANT
---------
Do NOT regenerate the 10 scenarios.
Do NOT rebuild the SPH pipeline.
Do NOT rebuild the D-Flow FM pipeline.
Do NOT recreate the FNO dataset.

Those artifacts already exist in the project backup.
"@ | Set-Content "$Checkpoint\PROJECT_STATE.txt" -Encoding utf8

# ------------------------------------------------------------
# 5. Important external source archives
# ------------------------------------------------------------

$DualSPHZip = "C:\Users\shash\OneDrive\Desktop\JAL RAKSHA\DualSPHysics_v5.4.3.zip"

if (Test-Path $DualSPHZip) {
    Copy-Item $DualSPHZip "$Checkpoint\external_sources\" -Force
}

# ------------------------------------------------------------
# 6. Installed-software paths
# ------------------------------------------------------------

@"
DualSPHysics
-----------
C:\Users\shash\OneDrive\Desktop\JAL RAKSHA\DualSPHysics_v5.4.3\DualSPHysics_v5.4

D-Flow FM
---------
C:\Program Files\Deltares\Delft3D FM Suite 2026.01 HMWQ

D-Flow FM CLI
-------------
C:\Program Files\Deltares\Delft3D FM Suite 2026.01 HMWQ\plugins\DeltaShell.Dimr\kernels\x64\bin\dflowfm-cli.exe

D-Flow FM launcher
------------------
C:\Program Files\Deltares\Delft3D FM Suite 2026.01 HMWQ\bin\run_dflowfm.bat

Miniconda
---------
C:\Users\shash\miniconda3

Conda environment
-----------------
C:\Users\shash\miniconda3\envs\jalraksha
"@ | Set-Content "$Checkpoint\SOFTWARE_PATHS.txt" -Encoding utf8

# ------------------------------------------------------------
# 7. PowerShell session history
# ------------------------------------------------------------

try {
    Get-History |
        Format-List -Property Id, CommandLine |
        Out-File "$Checkpoint\powershell_history.txt" -Encoding utf8
}
catch {
    "PowerShell history unavailable." |
        Out-File "$Checkpoint\powershell_history.txt" -Encoding utf8
}

# ------------------------------------------------------------
# 8. Complete project ZIP
# ------------------------------------------------------------

$ZipPath = Join-Path $BackupRoot "JalRaksha_Project_$Stamp.zip"

Write-Host ""
Write-Host "Creating complete project ZIP..."
Compress-Archive -Path "$Root\*" -DestinationPath $ZipPath -Force

# ------------------------------------------------------------
# 9. Backup summary
# ------------------------------------------------------------

@"
JALRAKSHA BACKUP COMPLETE

Checkpoint directory:
$Checkpoint

Complete project ZIP:
$ZipPath

The project source, simulations, outputs, FNO dataset, scripts,
environment information, VS Code extension list, and project state
have been captured.

NEXT TASK:
FNO training and validation.
"@ | Set-Content "$Checkpoint\BACKUP_SUMMARY.txt" -Encoding utf8

Write-Host ""
Write-Host "============================================================"
Write-Host "BACKUP COMPLETE"
Write-Host "============================================================"
Write-Host "Checkpoint:"
Write-Host $Checkpoint
Write-Host ""
Write-Host "Project ZIP:"
Write-Host $ZipPath
Write-Host ""
Write-Host "Your current JalRaksha state is saved."