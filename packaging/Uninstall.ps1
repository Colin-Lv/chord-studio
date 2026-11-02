param([Parameter(Mandatory=$true)][string]$InstallDir, [switch]$ConfirmUninstall)
$ErrorActionPreference = 'Stop'
$appPath = [IO.Path]::GetFullPath($InstallDir).TrimEnd('\')
$recordFile = Join-Path $appPath 'setup-owner.json'
$record = Get-Content -LiteralPath $recordFile -Raw -Encoding UTF8 | ConvertFrom-Json
if ($record.application -ne 'Chord Studio' -or $record.home -ne $appPath) { throw '安装目录记录不匹配，未删除任何文件。' }
if ($appPath -eq [IO.Path]::GetPathRoot($appPath).TrimEnd('\') -or $appPath -eq $env:USERPROFILE) { throw '不能卸载此目录。' }
$dataPath = [IO.Path]::GetFullPath($record.data_dir).TrimEnd('\')
if (-not $ConfirmUninstall) {
    Add-Type -AssemblyName System.Windows.Forms
    $answer=[Windows.Forms.MessageBox]::Show("卸载 Chord Studio 程序和运行环境？`n你的原曲、项目和模型会保留在：`n$dataPath",'Chord Studio','YesNo','Question')
    if ($answer -ne 'Yes') { exit 0 }
}
$created=$false
$mutexName='Local\ChordStudio-' + $dataPath.Replace('\','_').Replace('/','_').Replace(':','_').ToLowerInvariant()
$mutex=New-Object Threading.Mutex($false,$mutexName,[ref]$created)
try {
    if (-not $created) { throw '请先关闭 Chord Studio；正在运行的音频任务不会被强制停止。' }
    $names=@('backend','runtime','assets','setup','.setup-cache','.backups','chord-studio.exe','installation.json','setup-owner.json','.setup.lock','runtime-install.log','setup-install.log','setup-install-errors.log','setup-python.log','setup-python-errors.log','setup-webview.log','setup-webview-errors.log')
    $targets=@()
    foreach($name in $names) {
        $target=[IO.Path]::GetFullPath((Join-Path $appPath $name))
        if (-not $target.StartsWith($appPath+'\',[StringComparison]::OrdinalIgnoreCase)) { throw '卸载范围校验失败。' }
        if ($target -eq $dataPath -or $dataPath.StartsWith($target+'\',[StringComparison]::OrdinalIgnoreCase)) { throw '项目目录位于卸载范围内，操作已停止。' }
        if (Test-Path -LiteralPath $target) {
            $item=Get-Item -LiteralPath $target -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw '程序目录包含外部链接，请先处理链接。' }
            # Check every descendant before recursive removal; never follow junctions.
            if ($item.PSIsContainer) {
                $links=Get-ChildItem -LiteralPath $target -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint } | Select-Object -First 1
                if ($links) { throw '程序目录包含外部链接，未执行卸载。' }
            }
            $targets+=$target
        }
    }
    # All absolute paths and data exclusions have been checked before any delete.
    $shell=New-Object -ComObject WScript.Shell
    $link=Join-Path ([Environment]::GetFolderPath('Programs')) 'Chord Studio.lnk'
    if(Test-Path -LiteralPath $link) {
        $shortcut=$shell.CreateShortcut($link)
        if($shortcut.TargetPath -eq (Join-Path $appPath 'chord-studio.exe')) { Remove-Item -LiteralPath $link -Force }
    }
    $sha=[Security.Cryptography.SHA256]::Create()
    try {$id=([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($appPath.ToLowerInvariant())))).Replace('-','').Substring(0,16)} finally {$sha.Dispose()}
    $key='HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\ChordStudio-'+$id
    if(Test-Path -LiteralPath $key){Remove-Item -LiteralPath $key -Recurse}
    foreach($target in $targets){Remove-Item -LiteralPath $target -Recurse -Force}
    Write-Output "已卸载程序。项目和模型已保留：$dataPath"
} finally { $mutex.Dispose() }
