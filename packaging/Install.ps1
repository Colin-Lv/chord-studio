param(
    [switch]$Headless,
    [string]$InstallDir,
    [string]$DataDir,
    [ValidateSet('core','recommended','all')][string]$Preset = 'recommended',
    [switch]$NoShortcuts,
    [string]$Wheelhouse,
    [string]$ToolCache
)
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$releaseRoot = Split-Path -Parent $PSScriptRoot
$utf8 = New-Object System.Text.UTF8Encoding($false)
. (Join-Path $PSScriptRoot 'Prerequisites.ps1')

function Resolve-InstallLayout([string]$AppDirectory, [string]$ProjectDirectory) {
    $appPath = [IO.Path]::GetFullPath($AppDirectory).TrimEnd('\')
    if (-not $ProjectDirectory) { $ProjectDirectory = Join-Path $appPath 'user-data' }
    $projectPath = [IO.Path]::GetFullPath($ProjectDirectory).TrimEnd('\')
    if ($appPath -eq [IO.Path]::GetPathRoot($appPath).TrimEnd('\') -or $appPath -eq $env:USERPROFILE) {
        throw '请选择独立的 Chord Studio 安装文件夹。'
    }
    if ($projectPath -eq $appPath -or $appPath.StartsWith($projectPath + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw '项目目录不能包含程序目录。'
    }
    foreach ($part in @('backend','runtime','assets','setup','.setup-cache','.backups')) {
        $reserved = Join-Path $appPath $part
        if ($projectPath -eq $reserved -or $projectPath.StartsWith($reserved + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw '项目目录不能位于程序或运行环境文件夹内。'
        }
    }
    $marker = Join-Path $appPath 'setup-owner.json'
    if ((Test-Path -LiteralPath $appPath) -and @(Get-ChildItem -LiteralPath $appPath -Force).Count) {
        if (-not (Test-Path -LiteralPath $marker)) { throw '此目录包含其他文件。请选择空目录，原有项目无需删除。' }
        $owner = Get-Content -LiteralPath $marker -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($owner.application -ne 'Chord Studio' -or $owner.home -ne $appPath -or $owner.data_dir -ne $projectPath) {
            throw '安装目录或项目目录与原记录不同，请保留原目录后重试。'
        }
    }
    if ($appPath -eq $releaseRoot -or $appPath.StartsWith($releaseRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or $releaseRoot.StartsWith($appPath+'\',[StringComparison]::OrdinalIgnoreCase)) {
        throw '请把程序安装到解压后的安装包文件夹之外。'
    }
    [IO.Directory]::CreateDirectory($appPath) | Out-Null
    @{schema=1;application='Chord Studio';home=$appPath;data_dir=$projectPath} | ConvertTo-Json | ForEach-Object { [IO.File]::WriteAllText($marker, $_, $utf8) }
    return @($appPath, $projectPath)
}

function Quote-Argument([string]$Value) {
    # Windows paths cannot contain quotes; reject rather than construct shell code.
    if ($Value.Contains('"') -or $Value.Contains("`n") -or $Value.Contains("`r")) { throw '路径包含不支持的字符。' }
    return '"' + $Value.TrimEnd('\') + '"'
}

function Start-CheckedProcess([string]$Exe, [string[]]$Arguments, [string]$LogName) {
    $outFile = Join-Path $script:appHome ($LogName + '.log')
    $errFile = Join-Path $script:appHome ($LogName + '-errors.log')
    $argumentLine = ($Arguments | ForEach-Object { Quote-Argument $_ }) -join ' '
    $process = Start-Process -FilePath $Exe -ArgumentList $argumentLine -WindowStyle Hidden -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile
    # Keep the process handle before polling; Windows PowerShell 5.1 otherwise
    # can lose ExitCode for a short-lived redirected process.
    $null = $process.Handle
    $script:activeProcess = $process
    while (-not $process.HasExited) {
        Start-Sleep -Milliseconds 180
        if (-not $Headless) {
            if (Test-Path -LiteralPath $outFile) {
                $last = Get-Content -LiteralPath $outFile -Tail 1 -Encoding UTF8 -ErrorAction SilentlyContinue
                if ($last) {
                    try {
                        $status = $last | ConvertFrom-Json
                        if ($status.message) { $statusLabel.Text = $status.message }
                        elseif ($status.file) {
                            $statusLabel.Text = $status.file
                            if ($status.total) { $statusLabel.Text += ('  {0:N0}%' -f (100 * $status.bytes / $status.total)) }
                        }
                    } catch { }
                }
            }
            [Windows.Forms.Application]::DoEvents()
        }
        $process.Refresh()
    }
    $process.WaitForExit()
    $script:activeProcess = $null
    if ($process.ExitCode -ne 0) {
        $detail = Get-Content -LiteralPath $outFile -Tail 1 -Encoding UTF8 -ErrorAction SilentlyContinue
        try { $detail = ($detail | ConvertFrom-Json).message } catch { }
        throw "安装未完成。$detail`n日志：$outFile`n重试时会复用已下载内容。"
    }
}

function Install-Selected([string]$AppDirectory, [string]$ProjectDirectory, [string[]]$Features, [string[]]$Packs) {
    if (-not [Environment]::Is64BitOperatingSystem) { throw '此版本需要 64 位 Windows 10 或 Windows 11。' }
    $layout = Resolve-InstallLayout $AppDirectory $ProjectDirectory
    $script:appHome = $layout[0]; $projectHome = $layout[1]
    $cache = Join-Path $script:appHome '.setup-cache'
    [IO.Directory]::CreateDirectory($cache) | Out-Null
    Install-SystemPrerequisites $cache
    $uvArchive = Join-Path $cache 'uv-0.13.6.zip'
    $uvHash = 'a2a8820d5eea32f08ef467ef3b7f66883a9b076d5737a2fc2fbb5f458d1bb50f'
    if (-not (Test-Path -LiteralPath $uvArchive) -or (Get-FileHash -LiteralPath $uvArchive -Algorithm SHA256).Hash -ne $uvHash) {
        if (-not $Headless) { $statusLabel.Text = '正在下载自动配置工具'; [Windows.Forms.Application]::DoEvents() }
        $web = New-Object Net.WebClient
        try { $web.DownloadFile('https://github.com/astral-sh/uv/releases/download/0.13.6/uv-x86_64-pc-windows-msvc.zip', $uvArchive + '.partial') }
        finally { $web.Dispose() }
        if ((Get-FileHash -LiteralPath ($uvArchive+'.partial') -Algorithm SHA256).Hash -ne $uvHash) { throw '配置工具下载校验失败，请重试。' }
        Move-Item -LiteralPath ($uvArchive+'.partial') -Destination $uvArchive -Force
    }
    $uvFolder = Join-Path $cache 'uv-tool'
    Expand-Archive -LiteralPath $uvArchive -DestinationPath $uvFolder -Force
    $uvExe = Join-Path $uvFolder 'uv.exe'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $script:appHome 'runtime/python'
    $env:UV_PYTHON_INSTALL_REGISTRY = 'false'
    $env:UV_PYTHON_BIN_DIR = Join-Path $script:appHome 'runtime/bin'
    $env:UV_CACHE_DIR = Join-Path $cache 'uv'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONNOUSERSITE = '1'
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    if (-not $Headless) { $statusLabel.Text = '正在准备应用专用运行环境' }
    Start-CheckedProcess $uvExe @('python','install','3.13.12','--no-registry','--no-bin') 'setup-python'
    $python = Join-Path $env:UV_PYTHON_INSTALL_DIR 'cpython-3.13.12-windows-x86_64-none/python.exe'
    $installArgs = @((Join-Path $PSScriptRoot 'install.py'),'--home',$script:appHome,'--data-dir',$projectHome,'--source',$releaseRoot,'--uv',$uvExe,'--features') + $Features + @('--packs') + $Packs
    if ($Wheelhouse) { $installArgs += @('--wheelhouse',$Wheelhouse) }
    if ($ToolCache) { $installArgs += @('--tool-cache',$ToolCache) }
    Start-CheckedProcess $python $installArgs 'setup-install'
    if (-not $NoShortcuts) {
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Programs')) 'Chord Studio.lnk'))
        $shortcut.TargetPath = Join-Path $script:appHome 'chord-studio.exe'
        $shortcut.WorkingDirectory = $script:appHome
        $shortcut.Description = 'Chord Studio — 和弦、分轨与 MIDI 工作台'
        $shortcut.Save()
        $sha=[Security.Cryptography.SHA256]::Create()
        try {$id=([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($script:appHome.ToLowerInvariant())))).Replace('-','').Substring(0,16)} finally {$sha.Dispose()}
        $key='HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\ChordStudio-'+$id
        New-Item -Path $key -Force | Out-Null
        $values=@{
            DisplayName='Chord Studio'; DisplayVersion='1.1.0'; Publisher='Colin Lv';
            InstallLocation=$script:appHome; DisplayIcon=(Join-Path $script:appHome 'chord-studio.exe');
            UninstallString=('powershell.exe -NoProfile -ExecutionPolicy Bypass -File '+(Quote-Argument (Join-Path $script:appHome 'setup/Uninstall.ps1'))+' -InstallDir '+(Quote-Argument $script:appHome));
            URLInfoAbout='https://github.com/Colin-Lv/chord-studio'
        }
        foreach($entry in $values.GetEnumerator()){New-ItemProperty -LiteralPath $key -Name $entry.Key -Value $entry.Value -PropertyType String -Force | Out-Null}
    }
    return $script:appHome
}

if (-not $InstallDir) { $InstallDir = Join-Path $env:LOCALAPPDATA 'Programs/Chord Studio' }
if ($Headless) {
    $features = @(); $packs = @()
    if ($Preset -ne 'core') { $features = @('gpu','piano'); $packs = @('strings_v2','guitar_timbre') }
    if ($Preset -eq 'all') {
        $features += 'reconstruction'
        $catalog = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'model-packs.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        $packs = @($catalog.packs.PSObject.Properties.Name)
    }
    Install-Selected $InstallDir $DataDir $features $packs
    exit 0
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[Windows.Forms.Application]::EnableVisualStyles()
$form = New-Object Windows.Forms.Form
$form.Text = '安装 Chord Studio'; $form.Size = New-Object Drawing.Size(720,690)
$form.StartPosition = 'CenterScreen'; $form.FormBorderStyle = 'FixedDialog'; $form.MaximizeBox = $false
$form.Font = New-Object Drawing.Font('Microsoft YaHei UI',10)
$form.BackColor = [Drawing.Color]::FromArgb(249,247,242)
function Add-Label([string]$Text,[int]$X,[int]$Y,[int]$Width,[int]$Height) {
    $label = New-Object Windows.Forms.Label
    $label.Text=$Text; $label.Location=New-Object Drawing.Point($X,$Y); $label.Size=New-Object Drawing.Size($Width,$Height)
    $form.Controls.Add($label); return $label
}
$title = Add-Label '把音乐带进自己的工作台' 28 22 650 40
$title.Font = New-Object Drawing.Font('Microsoft YaHei UI',19,[Drawing.FontStyle]::Bold)
$null = Add-Label '自动配置运行环境；模型按需安装。原曲与项目保留在你的电脑上。' 30 70 650 32
$null = Add-Label '程序目录' 30 115 90 26
$appBox=New-Object Windows.Forms.TextBox; $appBox.Text=$InstallDir; $appBox.SetBounds(120,112,490,30); $form.Controls.Add($appBox)
$browse=New-Object Windows.Forms.Button; $browse.Text='选择'; $browse.SetBounds(620,110,60,32); $form.Controls.Add($browse)
$browse.Add_Click({ $dialog=New-Object Windows.Forms.FolderBrowserDialog; if($dialog.ShowDialog() -eq 'OK') {$appBox.Text=Join-Path $dialog.SelectedPath 'Chord Studio'}; $dialog.Dispose() })
$null = Add-Label '项目目录' 30 158 90 26
$dataBox=New-Object Windows.Forms.TextBox; $dataBox.Text=$DataDir; $dataBox.SetBounds(120,155,560,30); $form.Controls.Add($dataBox)
$null = Add-Label '留空时使用程序目录中的 user-data；升级时保留项目目录。' 120 191 550 26
$null = Add-Label '基础功能始终安装：和弦识别、节拍与小节、播放工作台、任务队列' 30 232 650 30
$options = New-Object Windows.Forms.CheckedListBox
$options.SetBounds(30,268,650,252); $options.CheckOnClick=$true; $options.BorderStyle='FixedSingle'
$items = @(
    @('提琴组＋六轨＋木／电吉他细分（推荐）','default'),
    @('钢琴音频转 MIDI，含力度、时值与踏板','piano'),
    @('标准四轨分离','standard'),
    @('精细四轨分离（四模型）','fine'),
    @('提琴组另一版本','strings_v1'),
    @('主唱与背景和声细分','vocal_detail'),
    @('鼓组细分','drum_detail'),
    @('人声优先分离','vocal_priority'),
    @('旋律／伴奏吉他细分','guitar_roles'),
    @('整曲 MIDI 重构（实验功能，下载较大）','reconstruction')
)
foreach($item in $items) { $null=$options.Items.Add($item[0], $item[1] -in @('default','piano')) }
$form.Controls.Add($options)
$statusLabel=Add-Label '首次安装需要联网。下载中断后可重新运行，复用已验证的文件。' 30 535 650 48
$terms=New-Object Windows.Forms.LinkLabel; $terms.Text='查看组件来源与模型使用条款'; $terms.SetBounds(30,602,390,30); $form.Controls.Add($terms)
$terms.Add_LinkClicked({ Start-Process (Join-Path $releaseRoot 'THIRD_PARTY.md') })
$button=New-Object Windows.Forms.Button; $button.Text='安装 / 补充功能'; $button.SetBounds(470,590,210,44); $form.Controls.Add($button)
$script:installing=$false
$form.Add_FormClosing({param($sender,$event) if($script:installing){$event.Cancel=$true; $statusLabel.Text='安装正在进行，请等待完成；重试会保留下载断点。'}})
$button.Add_Click({
    if($script:installing){return}
    $script:installing=$true; $button.Enabled=$false; $options.Enabled=$false; $appBox.Enabled=$false; $dataBox.Enabled=$false; $browse.Enabled=$false
    try {
        $features=@();$packs=@()
        foreach($index in $options.CheckedIndices) {
            $key=$items[$index][1]
            if($key -eq 'default'){$features+='gpu';$packs+=@('strings_v2','guitar_timbre')}
            elseif($key -eq 'piano'){$features+='piano'}
            elseif($key -eq 'reconstruction'){$features+='reconstruction';$packs+='reconstruction'}
            else{$features+='gpu';$packs+=$key}
        }
        $installed=Install-Selected $appBox.Text $dataBox.Text @($features | Select-Object -Unique) @($packs | Select-Object -Unique)
        $statusLabel.Text="安装完成。可从开始菜单打开 Chord Studio。`n$installed"
        $button.Text='完成'; $script:installing=$false
        [Windows.Forms.MessageBox]::Show("安装完成。`n$installed",'Chord Studio','OK','Information') | Out-Null
    } catch {
        $statusLabel.Text=$_.Exception.Message
        [Windows.Forms.MessageBox]::Show($_.Exception.Message,'Chord Studio · 安装未完成','OK','Error') | Out-Null
    } finally {
        $script:installing=$false; $button.Enabled=$true; $options.Enabled=$true; $appBox.Enabled=$true; $dataBox.Enabled=$true; $browse.Enabled=$true
    }
})
$null=$form.ShowDialog()
