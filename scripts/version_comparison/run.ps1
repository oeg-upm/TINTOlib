param(
    [Parameter(Mandatory = $true)]
    [string]$PreviousPython,

    [Parameter(Mandatory = $true)]
    [string]$CurrentPython,

    [Parameter(Mandatory = $true)]
    [string]$Splits,

    [Parameter(Mandatory = $true)]
    [string]$Dataset,

    [Parameter(Mandatory = $true)]
    [ValidateSet("classification", "regression", "unsupervised")]
    [string]$Problem,

    [string]$PreviousLabel = "v1.0.6.1",
    [string]$OutputRoot = "comparison_results"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$generator = Join-Path $PSScriptRoot "generate.py"
$comparer = Join-Path $PSScriptRoot "compare.py"
$sharedConfig = Join-Path $repositoryRoot "configs\version_comparison\shared_methods.json"
$currentOnlyConfig = Join-Path $repositoryRoot "configs\version_comparison\current_only.json"
$splitPath = (Resolve-Path $Splits).Path
$datasetOutput = Join-Path (Join-Path $repositoryRoot $OutputRoot) $Dataset

& $PreviousPython $generator `
    --config $sharedConfig `
    --splits $splitPath `
    --output $datasetOutput `
    --problem $Problem `
    --label $PreviousLabel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $CurrentPython $generator `
    --config $sharedConfig `
    --splits $splitPath `
    --output $datasetOutput `
    --problem $Problem `
    --label "current"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $CurrentPython $comparer `
    --before (Join-Path $datasetOutput $PreviousLabel) `
    --after (Join-Path $datasetOutput "current") `
    --report (Join-Path $datasetOutput "comparison.json")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $CurrentPython $generator `
    --config $currentOnlyConfig `
    --splits $splitPath `
    --output $datasetOutput `
    --problem $Problem `
    --label "current-only"
exit $LASTEXITCODE
