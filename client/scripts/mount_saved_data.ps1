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
# NOTE: Do not force BatchMode=yes; allow password prompt fallback when key auth is not available.
$sshArgs = @(
    # Avoid reading ~/.ssh/config (Windows backslashes in IdentityFile often break parsing)
    "-F", "NUL",
    "-o", "StrictHostKeyChecking=accept-new"
)
if ($sshKey) { $sshArgs += @("-i", $sshKey) }

$remoteSaved = "$basePath/server/saved_data"
 
$rsyncCmd = Get-Command "rsync.exe" -ErrorAction SilentlyContinue -All | Select-Object -First 1

function Invoke-ScpSync {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath,
        [Parameter(Mandatory = $true)][string]$RemoteUser,
        [Parameter(Mandatory = $true)][string]$RemoteHost,
        [Parameter(Mandatory = $true)][string[]]$SshArgs
    )

    $scpCmdInfo = Get-Command "scp.exe" -ErrorAction SilentlyContinue -All | Select-Object -First 1
    if (-not $scpCmdInfo) { throw "scp not found. Install OpenSSH client." }

    # mimic rsync --delete by clearing destination before copying contents
    Get-ChildItem -Force $DestinationPath -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

    # copy contents of saved_data (not the directory itself) to keep structure consistent
    $rsrc = $RemoteUser + "@" + $RemoteHost + ":`"$SourcePath/*`""
    Write-Host "-> scp -r $rsrc -> $DestinationPath (dst cleared before copy)"
    & $scpCmdInfo.Source @($SshArgs) "-r" $rsrc $DestinationPath

    if ($LASTEXITCODE -ne 0) { throw "scp failed with exit code $LASTEXITCODE" }
}

if ($rsyncCmd) {
    $rsrc = $remoteUser + "@" + $remoteHost + ":" + $remoteSaved + "/"
    Write-Host "-> rsync $rsrc -> $dst"
    $sshCmd = "ssh " + ((@($sshArgs) | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' ')
    & $rsyncCmd.Source "-az" "--delete" "-e" $sshCmd $rsrc $dst

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "rsync failed with exit code $LASTEXITCODE; falling back to scp (slower). If this keeps happening, install rsync on the server or remove login banner output in shell rc files."
        Invoke-ScpSync -SourcePath $remoteSaved -DestinationPath $dst -RemoteUser $remoteUser -RemoteHost $remoteHost -SshArgs $sshArgs
    }
} else {
    Invoke-ScpSync -SourcePath $remoteSaved -DestinationPath $dst -RemoteUser $remoteUser -RemoteHost $remoteHost -SshArgs $sshArgs
}
 
Write-Host "Done."
