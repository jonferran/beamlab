# Close every running BeamLab window NORMALLY (so it saves its settings), wait until
# each has exited, then start exactly one new BeamLab. Looks at ALL processes, not
# only "python" (pythonw / py-launched instances were missed before).
param([string]$AppArgs = '--connect')

$open = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like 'BeamLab*' }
foreach ($p in $open) {
    Write-Output ("closing BeamLab pid {0} ({1}) '{2}'" -f $p.Id, $p.ProcessName, $p.MainWindowTitle)
    [void]$p.CloseMainWindow()
    if (-not $p.WaitForExit(15000)) {
        Write-Output ("pid {0} did not close within 15 s - NOT starting a second one" -f $p.Id)
        exit 1
    }
}
if (-not $open) { Write-Output 'no BeamLab window was open' }
$left = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like 'BeamLab*' }
if ($left) { Write-Output 'a BeamLab window is still open - not starting another'; exit 1 }
Write-Output 'starting BeamLab'
python -m beamlab $AppArgs.Split(' ')
