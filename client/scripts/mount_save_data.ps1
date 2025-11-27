# PowerShell script: client/scripts/mount_save_data.ps1
# remote.yaml の remote ブロックを読み取り、
# ${USER}@${HOST}:${RPATH}/server/saved_data/ -> client/mnt/ を rsync で同期する

$ErrorActionPreference = "Stop"

# === スクリプト位置から各ディレクトリを解決 ===
# このファイル: <repo>/client/scripts/mount_save_data.ps1
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path       # .../client/scripts
$ClientRoot = Split-Path -Parent $ScriptDir                         # .../client

# 設定ファイル・出力ディレクトリ
$Config = Join-Path $ClientRoot "config/remote.yaml"
$RDST   = Join-Path $ClientRoot "mnt"

if (-not (Test-Path $Config)) {
    Write-Error "Config not found: $Config"
    exit 1
}

# === remote.yaml から remote ブロックだけを抽出（コメント # を除去） ===
$lines = Get-Content -Path $Config -Encoding UTF8

$startIdx = $null
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^\s*remote:') {
        $startIdx = $i
        break
    }
}

if ($null -eq $startIdx) {
    Write-Error "remote: ブロックが remote.yaml に見つかりません。"
    exit 1
}

$remoteLines = New-Object System.Collections.Generic.List[string]

for ($i = $startIdx; $i -lt $lines.Count; $i++) {
    $line = $lines[$i]

    # 1 行目以降で、先頭が非空白の行にぶつかったら remote ブロック終了
    if ($i -gt $startIdx -and $line -match '^[^\s]') {
        break
    }

    # コメント削除
    $lineNoComment = $line -replace '#.*$', ''
    $remoteLines.Add($lineNoComment)
}

function Get-YamlVal([string]$key) {
    foreach ($l in $remoteLines) {
        if ($l -match "^\s*$key\s*:\s*(.+)\s*$") {
            # 両端の空白とクォートを削除
            $v = $Matches[1].Trim()
            $v = $v.Trim('"').Trim("'")
            return $v
        }
    }
    return ""
}

$HOST  = Get-YamlVal "host"
$USER  = Get-YamlVal "user"
$KEY   = Get-YamlVal "ssh_key"
$RPATH = Get-YamlVal "path"

if ([string]::IsNullOrWhiteSpace($HOST) -or
    [string]::IsNullOrWhiteSpace($USER) -or
    [string]::IsNullOrWhiteSpace($RPATH)) {

    # ← ここは {} を含むので必ずシングルクォートにする
    Write-Error 'remote.yaml の remote: { host, user, path } を確認してください。'
    exit 1
}

# 鍵ファイルは任意。未指定なら ssh の通常設定(~/.ssh/config 等)に任せる
$SSH_OPT = @('-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes')

if (-not [string]::IsNullOrWhiteSpace($KEY) -and
    $KEY -ne "null" -and
    (Test-Path $KEY)) {

    $SSH_OPT = @('-i', $KEY) + $SSH_OPT
}

# === rsync のソース/宛先設定 ===
# 末尾のスラッシュが「中身だけ」を同期するポイント
$RPATH_CLEAN = $RPATH.TrimEnd('/', '\')

# ここは -f で文字列組み立てして、$HOST: の誤解釈を避ける
$RSRC = "{0}@{1}:{2}/server/saved_data/" -f $USER, $HOST, $RPATH_CLEAN

if (-not (Test-Path $RDST)) {
    New-Item -ItemType Directory -Path $RDST | Out-Null
}

Write-Host "-> Syncing ONLY $RSRC -> $RDST"

$sshOptString = $SSH_OPT -join ' '
# rsync -az --delete -e "ssh ${SSH_OPT[*]}" "$RSRC" "$RDST"
& rsync -az --delete -e "ssh $sshOptString" "$RSRC" "$RDST"

Write-Host "Done. (synced contents into $RDST)"