# Camera driver packages (not in this repository)

The installer's optional "Camera drivers" feature bundles the two USB driver
packages for the LBA-C-100's camera. They are the camera makers' software under
their own licences, so they are not committed here. Put them in this folder
before running `installer\build.ps1`:

| File | Maker | Where to get it |
|---|---|---|
| `pylon_USB_Camera_Driver.msi` | Basler AG | pylon USB camera driver (from the Basler pylon Camera Software Suite, or the LBA-C-100 software package's `_Drivers` folder) |
| `ic4-gentl-driver-usb3vision_1.6.0.894_Win_all.msi` | The Imaging Source Europe GmbH | IC4 GenTL Driver for USB3Vision Devices (theimagingsource.com downloads, or the LBA-C-100 software package's `_Drivers` folder) |

Different versions work too; update the file names in `installer\BeamLab.wxs`
and `installer\build.ps1` to match.
