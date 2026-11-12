# Build BeamLab-<version>.msi:  powershell -ExecutionPolicy Bypass -File installer\build.ps1
# Needs: Python 3.9 with pyinstaller, pypylon, imagingcontrol4, numpy, pillow;
#        WiX 3.14 binaries in %USERPROFILE%\tools\wix314 (or set -Wix);
#        the two vendor driver MSIs in installer\drivers\ (see installer\drivers\README.md).
param(
    [string]$Version = '0.1.4',
    [string]$Wix = "$env:USERPROFILE\tools\wix314"
)
$ErrorActionPreference = 'Stop'
$inst = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $inst
$build = Join-Path $root 'build'
$dist = Join-Path $root 'dist'
Set-Location $root

foreach ($f in 'drivers\pylon_USB_Camera_Driver.msi', 'drivers\ic4-gentl-driver-usb3vision_1.6.0.894_Win_all.msi') {
    if (-not (Test-Path (Join-Path $inst $f))) { throw "missing $f (see installer\drivers\README.md)" }
}
foreach ($t in 'heat.exe', 'candle.exe', 'light.exe') {
    if (-not (Test-Path (Join-Path $Wix $t))) { throw "WiX not found in $Wix" }
}

Write-Output '== 1/4 freezing the app (PyInstaller)'
if (Test-Path "$dist\BeamLab") { Remove-Item -Recurse -Force "$dist\BeamLab" }
python -m PyInstaller --noconfirm --clean --log-level WARN --workpath "$build\pyi" --distpath $dist "$inst\BeamLab.spec"
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed' }

Write-Output '== 2/4 self-test of the frozen app'
$st = Join-Path $build 'selftest_exe.txt'
if (Test-Path $st) { Remove-Item $st }
$p = Start-Process -FilePath "$dist\BeamLab\BeamLab.exe" -ArgumentList @('--selftest', "`"$st`"") -Wait -PassThru
Get-Content $st
if ($p.ExitCode -ne 0) { throw 'frozen app self-test failed' }

Write-Output '== 3/4 collecting files (heat)'
$wixobj = Join-Path $build 'wix'
New-Item -ItemType Directory -Force $wixobj | Out-Null
& "$Wix\heat.exe" dir "$dist\BeamLab" -nologo -cg AppFiles -dr INSTALLFOLDER -srd -sreg -scom -ag -var var.AppDir -platform x64 -out "$wixobj\AppFiles.wxs"
if ($LASTEXITCODE -ne 0) { throw 'heat failed' }

Write-Output '== 4/4 compiling the installer (candle + light)'
& "$Wix\candle.exe" -nologo -arch x64 "-dProductVersion=$Version" "-dAppDir=$dist\BeamLab" "-dInstallerDir=$inst" `
    -out "$wixobj\\" "$inst\BeamLab.wxs" "$wixobj\AppFiles.wxs" -ext WixUIExtension
if ($LASTEXITCODE -ne 0) { throw 'candle failed' }
$msi = Join-Path $dist "BeamLab-$Version.msi"
& "$Wix\light.exe" -nologo -ext WixUIExtension -cultures:en-us -out $msi "$wixobj\BeamLab.wixobj" "$wixobj\AppFiles.wixobj"
if ($LASTEXITCODE -ne 0) { throw 'light failed' }
Get-Item $msi | Select-Object FullName, @{n='MB'; e={[math]::Round($_.Length / 1MB, 1)}}
