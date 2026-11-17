# Windows launcher for run_all.sh: finds Git Bash from git.exe, because plain `bash` in PowerShell is
# the WSL launcher, which fails when no Linux distribution is installed.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 TotalFootball
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 TotalFootball -SkipVideo   # reuse detections
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -WhichBash                  # print the bash used
#
# (-ExecutionPolicy Bypass only for this one launch: Windows blocks .ps1 files by default.)
param([string]$Name = "TotalFootball", [switch]$SkipVideo, [switch]$WhichBash)

$git = (Get-Command git -ErrorAction Stop).Source
$bash = $null
for ($dir = Split-Path $git; $dir -and -not $bash; $dir = Split-Path $dir) {      # E:\Git\cmd, E:\Git\mingw64\bin, ...
    $bash = @("$dir\bin\bash.exe", "$dir\usr\bin\bash.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $bash) { throw "Git Bash not found above $git. Install Git for Windows, or run scripts/run_all.sh from any bash." }
if ($WhichBash) { Write-Output $bash; return }

Set-Location (Split-Path $PSScriptRoot)
if ($SkipVideo) { $env:SKIP_VIDEO = "1" }
$env:Path = "$(Split-Path $bash);$env:Path"                # so ffmpeg and friends installed for Git are found
& $bash scripts/run_all.sh $Name
exit $LASTEXITCODE
