"""Entry point for the frozen BeamLab.exe (PyInstaller). main() calls
multiprocessing.freeze_support() first, so the camera process (spawned by
re-running this exe) works in the frozen build."""
from beamlab.app import main

if __name__ == '__main__':
    main()
