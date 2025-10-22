-- MPR Viewer

A complete **Multi-Planar Reconstruction (MPR)** viewer for medical imaging.

-- Features

-  View medical volumes in (Axial), (Coronal), and (Sagittal) planes.  
- ROI-based navigation — limit scrolling within defined regions.  
- Save cropped volumes with accurate orientation as `.nii.gz`.  
- Oblique plane visualization for arbitrary slice orientations.  
-  Compatible with both DICOM series and NIfTI files.  
-  Interactive UI built using PyQt5.

-- Requirements

Install dependencies using pip:

```
pip install nibabel numpy PyQt5 Pillow scipy pydicom
```
This is a MPR viewer that views the DICOM series or NIFTI files in 3 views Axial,Sagittal,Coronal and oblique
and it gives the user the option to go through the slices with a slider and the crosshair in the other 3 planes will move with it
and ROI.

