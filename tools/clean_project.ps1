param(
    [switch]$DryRun,
    [switch]$Yes,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\', '/')
$rootPrefix = $projectRoot + [IO.Path]::DirectorySeparatorChar

if ($Help) {
    Write-Output 'clean.bat [--dry-run] [--yes]'
    Write-Output 'Removes project-local models, browsers, storage, dependencies, builds, caches and .env.'
    Write-Output 'Preserves source, .env.example, lockfiles, examples source/assets, .git, .agents and .codex.'
    exit 0
}

# Derive the root from this file, never from the caller working directory.
foreach ($marker in @('setup.bat', '.env.example', 'app\config.py', 'README.md')) {
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $marker) -PathType Leaf)) {
        throw "Project marker missing: $marker. Refusing cleanup."
    }
}
if ((Get-Item -LiteralPath $projectRoot -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw 'Refusing cleanup through a linked project directory.'
}

function Assert-LocalPath([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if (-not $absolute.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup target escapes the project: $absolute"
    }
    # Reject redirected parent directories as well as redirected targets.
    $current = $absolute
    while ($current -ne $projectRoot) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if ($item.PSIsContainer -and ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                throw "Linked directory cannot be cleaned safely: $current"
            }
        }
        $current = [IO.Path]::GetDirectoryName($current)
    }
    return $absolute
}

$targets = New-Object 'System.Collections.Generic.List[string]'
function Add-Target([string]$Path) {
    $absolute = Assert-LocalPath $Path
    if (Test-Path -LiteralPath $absolute) {
        if (-not $targets.Contains($absolute)) { $targets.Add($absolute) }
    }
}

foreach ($name in @('.venv', 'omniui_env', 'models_cache', 'storage', '.env',
    '.pytest_cache', '.mypy_cache', '.ruff_cache', '.cache', 'htmlcov',
    '.coverage', 'coverage.xml', 'node_modules', 'dist', 'build')) {
    Add-Target (Join-Path $projectRoot $name)
}

# Walk source directories without following junctions or descending into large
# trees already selected for deletion. Never traverse repository/tool metadata.
function Find-Artifacts([string]$Directory) {
    foreach ($item in Get-ChildItem -LiteralPath $Directory -Force) {
        if ($item.Name -in @('.git', '.agents', '.codex')) { continue }
        if ($targets.Contains($item.FullName)) { continue }
        if ($item.PSIsContainer) {
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
            if ($item.Name -in @('__pycache__', 'node_modules', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.vite') -or
                ($Directory.StartsWith((Join-Path $projectRoot 'examples'), [StringComparison]::OrdinalIgnoreCase) -and $item.Name -in @('dist', 'build'))) {
                Add-Target $item.FullName
            } else { Find-Artifacts $item.FullName }
        } elseif ($item.Extension -in @('.pyc', '.pyo') -or
            ($Directory -eq $projectRoot -and ($item.Name -like 'yolov*.pt' -or $item.Name -like 'yolov*.onnx' -or
                $item.Name -like 'omniui_*.zip' -or $item.Extension -eq '.log'))) {
            Add-Target $item.FullName
        }
    }
}
Find-Artifacts $projectRoot

Write-Output "OmniUI fresh-setup cleanup: $projectRoot"
Write-Output 'Deletes uploads, jobs, exports, logs, all cached models (including LTX), browsers, dependencies and .env.'
Write-Output 'Source code, Git history, example source/assets and .env.example are preserved.'
foreach ($target in $targets) { Write-Output ('  ' + $target.Substring($rootPrefix.Length)) }

if ($DryRun) {
    Write-Output 'DRY RUN: nothing was deleted.'
    exit 0
}
if (-not $Yes) {
    $answer = Read-Host 'Permanently delete these local files? Type YES to continue'
    if ($answer.Trim() -ine 'YES') { Write-Output 'Cleanup cancelled.'; exit 0 }
}

# Never kill unrelated Python/Node applications. Stop the project's own service
# before resetting its environment and job records.
$running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe' OR Name='node.exe'" |
    Where-Object { $_.ProcessId -ne $PID -and (
        ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) -or
        ($_.CommandLine -and $_.CommandLine.IndexOf($rootPrefix, [StringComparison]::OrdinalIgnoreCase) -ge 0)) })
if ($running.Count -gt 0) {
    Write-Output 'Cleanup blocked: project processes are running. Nothing has been deleted.'
    foreach ($process in $running) {
        Write-Output ("  PID {0}: {1}" -f $process.ProcessId, $process.Name)
    }
    Write-Output 'Stop these processes in their terminals with Ctrl+C, then rerun clean.bat.'
    Write-Output 'For a background process, inspect it with Get-Process -Id <PID> and stop it with Stop-Process -Id <PID>.'
    exit 1
}

# Check every target and child directory before deleting anything. A linked
# directory is refused rather than traversed into another workspace/cache.
function Check-Tree([string]$Path) {
    $null = Assert-LocalPath $Path
    if (Test-Path -LiteralPath $Path -PathType Container) {
        foreach ($child in Get-ChildItem -LiteralPath $Path -Force) {
            if ($child.PSIsContainer) { Check-Tree $child.FullName }
        }
    }
}
foreach ($target in $targets) { Check-Tree $target }

$failures = New-Object 'System.Collections.Generic.List[string]'
foreach ($target in $targets) {
    try {
        $null = Assert-LocalPath $target
        Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction Stop
        Write-Output ('Removed: ' + $target.Substring($rootPrefix.Length))
    } catch {
        $failures.Add($target)
        Write-Warning ("Could not remove ${target}: " + $_.Exception.Message)
    }
}

# Restore only empty runtime scaffolding; setup recreates the environment.
foreach ($relative in @('models_cache', 'storage\uploads', 'storage\jobs', 'storage\outputs')) {
    $directory = Join-Path $projectRoot $relative
    $null = Assert-LocalPath $directory
    $null = New-Item -ItemType Directory -Path $directory -Force
    $placeholder = Join-Path $directory '.gitkeep'
    if (-not (Test-Path -LiteralPath $placeholder)) {
        $null = New-Item -ItemType File -Path $placeholder
    }
}
if ($failures.Count -gt 0) {
    Write-Output 'Cleanup incomplete. Close processes holding these files, then rerun clean.bat.'
    exit 1
}
Write-Output 'Cleanup complete. Run .\setup.bat for a fresh installation, then .\start.bat.'
Write-Output 'Only project-local files were cleaned; shared system/user caches were left alone.'
exit 0
