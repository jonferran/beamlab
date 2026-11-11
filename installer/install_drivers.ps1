# Installs the LBA camera USB drivers (Basler pylon USB + The Imaging Source USB3 Vision).
# Started by the BeamLab installer right AFTER it finishes (Windows Installer does not
# allow one .msi to run another while it is installing), so it first waits for that
# install to release the Windows Installer lock, then asks for admin rights once.
param([switch]$Elevated)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $env:ProgramData 'BeamLab'
$log = Join-Path $logDir 'driver_install.log'

function Show-Box($text, $title, $icon) {
    Add-Type -AssemblyName System.Windows.Forms
    [void][System.Windows.Forms.MessageBox]::Show($text, $title, 'OK', $icon)
}

if (-not $Elevated) {
    # 1) wait (up to 3 min) for the BeamLab install to finish and free the installer lock
    $deadline = (Get-Date).AddMinutes(3)
    while ((Get-Date) -lt $deadline) {
        $busy = $false
        try {
            $m = [System.Threading.Mutex]::OpenExisting('Global\_MSIExecute')
            $m.Dispose()
            $busy = $true
        } catch { $busy = $false }
        if (-not $busy) { break }
        Start-Sleep -Milliseconds 500
    }
    # 2) re-run this script elevated (one UAC prompt for both drivers)
    try {
        Start-Process -FilePath 'powershell.exe' -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$($MyInvocation.MyCommand.Path)`"", '-Elevated')
    } catch {
        Show-Box ("The camera drivers were NOT installed (administrator permission was declined).`n`n" +
                  "Fix: run this again as administrator:`n$($MyInvocation.MyCommand.Path)`n`n" +
                  "or use 'Install camera drivers' in the BeamLab Start menu folder.") 'BeamLab - camera drivers' 'Warning'
    }
    exit 0
}

# ---- elevated part: install both driver packages
New-Item -ItemType Directory -Force $logDir | Out-Null
"==== $(Get-Date -Format s) installing camera drivers" | Out-File -Append -Encoding utf8 $log
$results = @()
foreach ($msi in Get-ChildItem -Path $here -Filter '*.msi' | Sort-Object Name) {
    $msiLog = Join-Path $logDir ($msi.BaseName + '.log')
    $p = Start-Process -FilePath 'msiexec.exe' -Wait -PassThru -ArgumentList @(
        '/i', "`"$($msi.FullName)`"", '/passive', '/norestart', '/l*v', "`"$msiLog`"")
    $code = $p.ExitCode
    # 0 = done, 3010 = done (restart needed), 1638 = a same/newer version is already installed
    $ok = @(0, 3010, 1638) -contains $code
    "$($msi.Name): exit $code ($(if ($ok) {'OK'} else {'FAILED'}))" | Out-File -Append -Encoding utf8 $log
    $results += [pscustomobject]@{ Name = $msi.Name; Code = $code; Ok = $ok }
}
$failed = @($results | Where-Object { -not $_.Ok })
if ($failed.Count -gt 0) {
    $list = ($failed | ForEach-Object { "$($_.Name) (exit code $($_.Code))" }) -join "`n"
    Show-Box ("These camera drivers failed to install:`n$list`n`nDetails: $logDir`n`n" +
              "Fix: plug the LBA out, then use 'Install camera drivers' in the BeamLab Start menu folder.") 'BeamLab - camera drivers' 'Error'
    exit 1
}
if ($results | Where-Object { $_.Code -eq 3010 }) {
    Show-Box 'Camera drivers installed. Restart Windows before using the LBA.' 'BeamLab - camera drivers' 'Information'
}
exit 0
