# Microsoft-maintained shared prerequisites. Existing installations are reused.
function Test-WebViewRuntime {
    foreach ($key in @(
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}',
        'HKCU:\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'
    )) {
        $record = Get-ItemProperty -LiteralPath $key -ErrorAction SilentlyContinue
        if ($record -and $record.pv) {
            try { if ([version]$record.pv -gt [version]'0.0.0.0') { return $true } } catch { }
        }
    }
    return $false
}

function Test-VisualCppRuntime {
    $record = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64' -ErrorAction SilentlyContinue
    if ($record -and $record.Installed -eq 1 -and $record.Version) {
        try { return [version]$record.Version.TrimStart('v') -ge [version]'14.50.0.0' } catch { }
    }
    return $false
}

function Get-MicrosoftInstaller([string]$Url, [string]$FileName, [string]$CacheDirectory) {
    $target = Join-Path $CacheDirectory $FileName
    $web = New-Object Net.WebClient
    try { $web.DownloadFile($Url, $target + '.partial') }
    finally { $web.Dispose() }
    # Evergreen endpoints intentionally change. Validate Microsoft's trusted
    # Authenticode signature rather than pretending a stale hash pins them.
    $signature = Get-AuthenticodeSignature -LiteralPath ($target + '.partial')
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch '(^|,\s*)O=Microsoft Corporation(,|$)') {
        throw '微软运行组件的数字签名校验失败，未执行下载文件。'
    }
    Move-Item -LiteralPath ($target + '.partial') -Destination $target -Force
    return $target
}

function Install-SystemPrerequisites([string]$CacheDirectory) {
    if (-not (Test-WebViewRuntime)) {
        $exe = Get-MicrosoftInstaller 'https://go.microsoft.com/fwlink/p/?LinkId=2124703' 'MicrosoftEdgeWebview2Setup.exe' $CacheDirectory
        Start-CheckedProcess $exe @('/silent','/install') 'setup-webview'
        if (-not (Test-WebViewRuntime)) { throw 'WebView2 尚未准备完成。请重新运行安装程序。' }
    }
    if (-not (Test-VisualCppRuntime)) {
        $exe = Get-MicrosoftInstaller 'https://aka.ms/vc14/vc_redist.x64.exe' 'vc_redist.x64.exe' $CacheDirectory
        # VC Runtime is machine-wide. Windows shows its own elevation prompt only
        # on machines that need it; do not elevate the rest of our installation.
        $process = Start-Process -FilePath $exe -ArgumentList '/install /quiet /norestart' -Verb RunAs -WindowStyle Hidden -Wait -PassThru
        if ($process.ExitCode -notin @(0,3010,1638) -or -not (Test-VisualCppRuntime)) {
            throw 'Visual C++ 运行组件未安装完成。请完成微软安装程序后重试。'
        }
    }
}
