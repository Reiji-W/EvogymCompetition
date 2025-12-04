# client/scripts/mount_saved_data.ps1
# Parse remote.yaml and sync saved_data via rsync or scp (Windows PowerShell)
$ErrorActionPreference = "Stop"

$ConfigPath = "client/config/remote.yaml"
if (-not (Test-Path $ConfigPath)) { throw "Config not found: $ConfigPath" }

# parse remote block (simple)
$remote = @{}
$inRemote = $false
Get-Content $ConfigPath | ForEach-Object {
    $line = $_
    $trim = $line.Trim()
    if ($trim -match '^\s*#' -or $trim -eq '') { return }
    if ($trim -match '^remote:') { $inRemote = $true; return }
    if ($inRemote) {
        if ($line -notmatch '^\s') { return }
        if ($trim -match '^(\w+):\s*(.+)?$') {
            $key = $matches[1]
            $val = $matches[2].Trim("`"", "'")
            if ($val -eq 'null') { $val = $null }
            $remote[$key] = $val
        }
    }
}

foreach ($k in @("host","user","path")) {
    if (-not $remote[$k]) { throw "remote.yaml missing remote:$k" }
}

$remoteHost = $remote["host"]
$remoteUser = $remote["user"]
$basePath = $remote["path"].TrimEnd(@('/','\'))
$sshKey = $remote["ssh_key"]

$dst = Join-Path "client" "mnt"
New-Item -ItemType Directory -Force -Path $dst | Out-Null

# ssh/rsync options
$sshArgs = @("-o","BatchMode=yes","-o","StrictHostKeyChecking=accept-new")
if ($sshKey) { $sshArgs += @("-i", $sshKey) }

$remoteSaved = "$basePath/server/saved_data"

$rsync = Get-Command "rsync.exe" -ErrorAction SilentlyContinue
if ($rsync) {
    $rsrc = $remoteUser + "@" + $remoteHost + ":" + $remoteSaved + "/"
    Write-Host "-> rsync $rsrc -> $dst"
    & $rsync.FullName "-az" "--delete" "-e" ("ssh " + ($sshArgs -join ' ')) $rsrc $dst
} else {
    $scp = Get-Command "scp.exe" -ErrorAction SilentlyContinue
    if (-not $scp) { throw "rsync or scp not found. Install OpenSSH client or rsync." }
    $rsrc = $remoteUser + "@" + $remoteHost + ":" + $remoteSaved
    Write-Host "-> scp -r $rsrc -> $dst (delete not mirrored)"
    $scpCmd = $scp.Source
    & $scpCmd @($sshArgs) "-r" $rsrc $dst
}

Write-Host "Done."
