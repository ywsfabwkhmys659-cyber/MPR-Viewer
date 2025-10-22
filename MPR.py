# MPR_advanced_complete.py
# Complete MPR Viewer with:
# 1. ROI-based navigation limits across views
# 2. Save cropped volume from any viewport with proper orientation
# 3. Oblique plane visualization
# Requirements: nibabel numpy PyQt5 Pillow scipy pydicom

import sys, os
import numpy as np
import nibabel as nib
import pydicom
from pydicom.filereader import dcmread
from pydicom.errors import InvalidDicomError
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image
from scipy.ndimage import map_coordinates

# Function to show file dialog and get path
def get_nifti_file():
    file_dialog = QtWidgets.QFileDialog()
    file_dialog.setNameFilter("NIfTI files (*.nii *.nii.gz);;All files (*.*)")
    if file_dialog.exec_():
        return file_dialog.selectedFiles()[0]
    return None

def get_dicom_folder():
    """Show dialog to select DICOM directory"""
    folder = QtWidgets.QFileDialog.getExistingDirectory(
        None, "Select DICOM Directory", "",
        QtWidgets.QFileDialog.ShowDirsOnly)
    return folder if folder else None

def load_medical_image(path):
    """Load medical image data from either NIfTI file or DICOM directory
    Args:
        path (str): Path to NIfTI file or DICOM directory
    Returns:
        tuple: (numpy array, affine matrix)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
        
    # Check if path is a file (NIfTI) or directory (DICOM)
    if os.path.isfile(path):
        # Assume NIfTI file
        try:
            nii = nib.load(path)
            arr = nii.get_fdata()
            if arr.ndim == 4:
                arr = arr[..., 0]
            return np.asarray(arr, dtype=np.float32), (nii.affine if hasattr(nii, "affine") else np.eye(4))
        except Exception as e:
            raise ValueError(f"Error loading NIfTI file: {str(e)}")
    else:
        # Assume DICOM directory
        try:
            return load_dicom_series(path)
        except Exception as e:
            raise ValueError(f"Error loading DICOM series: {str(e)}")

def load_nifti(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    nii = nib.load(path)
    arr = nii.get_fdata()
    if arr.ndim == 4:
        arr = arr[..., 0]
    return np.asarray(arr, dtype=np.float32), (nii.affine if hasattr(nii, "affine") else np.eye(4))

def is_dicom_file(filepath):
    """Check if a file is a valid DICOM file"""
    try:
        with open(filepath, 'rb') as f:
            # Check for DICOM magic number
            f.seek(128)
            magic = f.read(4)
            return magic == b'DICM'
    except:
        return False

def load_dicom_series(directory):
    """Load a DICOM series from a directory
    Returns: (volume_array, affine_matrix)"""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all DICOM files in directory
    dicom_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            if is_dicom_file(filepath):
                dicom_files.append(filepath)

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {directory}")

    # Read first file to get series info
    ref_dicom = dcmread(dicom_files[0])
    
    # Load all files in series
    slices = []
    positions = []
    for filepath in dicom_files:
        dcm = dcmread(filepath)
        # Check if slice belongs to same series
        if (dcm.SeriesInstanceUID == ref_dicom.SeriesInstanceUID):
            try:
                slices.append(dcm)
                # Get slice position
                pos = dcm.ImagePositionPatient
                positions.append((float(pos[2]), dcm))  # Use Z position for sorting
            except:
                continue

    # Sort slices by position
    positions.sort(key=lambda x: x[0])
    slices = [x[1] for x in positions]

    # Extract pixel arrays and stack them
    try:
        voxel_array = np.stack([s.pixel_array.astype(float) for s in slices])
    except:
        raise ValueError("Unable to stack slice pixel arrays")

    # Apply rescale slope and intercept if available
    try:
        slope = float(slices[0].RescaleSlope)
        intercept = float(slices[0].RescaleIntercept)
        voxel_array = voxel_array * slope + intercept
    except:
        pass  # No rescale needed

    # Calculate affine matrix
    try:
        ps = slices[0].PixelSpacing
        ss = slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]
        ipp = slices[0].ImagePositionPatient
        iop = slices[0].ImageOrientationPatient

        affine = np.eye(4)
        affine[0:3, 0] = [iop[0]*ps[0], iop[1]*ps[0], iop[2]*ps[0]]
        affine[0:3, 1] = [iop[3]*ps[1], iop[4]*ps[1], iop[5]*ps[1]]
        affine[0:3, 2] = [0, 0, ss]
        affine[0:3, 3] = [ipp[0], ipp[1], ipp[2]]
    except:
        # If unable to calculate affine, use identity
        affine = np.eye(4)

    # Transpose array to match NIfTI orientation (sagittal, coronal, axial)
    voxel_array = np.transpose(voxel_array, (1, 2, 0))
    
    return np.asarray(voxel_array, dtype=np.float32), affine

def rotation_matrix_from_angles(angles):
    """Create 3D rotation matrix from angles (rx, ry, rz) in degrees"""
    rx, ry, rz = np.radians(angles)
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def extract_oblique_slice(volume, center, normal, up_vector, size=(256, 256)):
    """Extract an oblique slice from volume"""
    normal = normal / np.linalg.norm(normal)
    up_vector = up_vector / np.linalg.norm(up_vector)
    
    right = np.cross(up_vector, normal)
    right = right / np.linalg.norm(right)
    up = np.cross(normal, right)
    
    h, w = size
    y_coords = np.linspace(-h/2, h/2, h)
    x_coords = np.linspace(-w/2, w/2, w)
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    positions = (center[:, None, None] + 
                 X[None, :, :] * right[:, None, None] + 
                 Y[None, :, :] * up[:, None, None])
    
    slice_data = map_coordinates(volume, positions, order=1, mode='constant', cval=0)
    
    return slice_data

# ---------------- MainWindow ----------------
class MainWindow(QtWidgets.QWidget):
    def __init__(self, file_path):
        super().__init__()
        self.volume, self.affine = load_medical_image(file_path)
        self.global_roi = None  # (start_x, start_y, start_z, end_x, end_y, end_z)
        self._build_ui()
        
    def _build_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        
        # Top row with three orthogonal views
        top_row = QtWidgets.QHBoxLayout()
        
        self.axial_view = SingleView(self.volume, self.affine, 'axial', 'Axial View', '#3498db')
        self.coronal_view = SingleView(self.volume, self.affine, 'coronal', 'Coronal View', '#2ecc71')
        self.sagittal_view = SingleView(self.volume, self.affine, 'sagittal', 'Sagittal View', '#e74c3c')
        
        top_row.addWidget(self.axial_view)
        top_row.addWidget(self.coronal_view)
        top_row.addWidget(self.sagittal_view)
        main_layout.addLayout(top_row)
        
        # Add global ROI save button
        save_frame = QtWidgets.QFrame()
        save_frame.setStyleSheet("background:#34495e; border-radius:6px; padding:10px;")
        save_layout = QtWidgets.QHBoxLayout(save_frame)
        
        self.roi_coords_label = QtWidgets.QLabel("No ROI Selected")
        self.roi_coords_label.setStyleSheet("color:white;")
        save_layout.addWidget(self.roi_coords_label)
        
        self.global_save_btn = QtWidgets.QPushButton("💾 Save ROI Volume")
        self.global_save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        self.global_save_btn.clicked.connect(self.save_roi_volume)
        save_layout.addWidget(self.global_save_btn)
        
        main_layout.addWidget(save_frame)
        
        # Bottom row with oblique view
        self.oblique_view = ObliqueView(self.volume, self.affine)
        main_layout.addWidget(self.oblique_view)
        
        # Connect signals
        self.axial_view.slice_changed.connect(self.on_slice_changed)
        self.coronal_view.slice_changed.connect(self.on_slice_changed)
        self.sagittal_view.slice_changed.connect(self.on_slice_changed)
        
        self.axial_view.roi_drawn.connect(self.on_roi_drawn)
        self.coronal_view.roi_drawn.connect(self.on_roi_drawn)
        self.sagittal_view.roi_drawn.connect(self.on_roi_drawn)
        
        self.axial_view.roi_cleared.connect(self.on_roi_cleared)
        self.coronal_view.roi_cleared.connect(self.on_roi_cleared)
        self.sagittal_view.roi_cleared.connect(self.on_roi_cleared)
        
        self.axial_view.save_view_requested.connect(self.save_view)
        self.coronal_view.save_view_requested.connect(self.save_view)
        self.sagittal_view.save_view_requested.connect(self.save_view)
        
        self.setWindowTitle("MPR Viewer")
        self.resize(1200, 800)

    def on_slice_changed(self, x, y, z):
        # Update crosshair in all views
        if self.sender() != self.axial_view:
            self.axial_view.index = z
            self.axial_view.cross_pos = [y, x]
            self.axial_view.update_image()
        
        if self.sender() != self.coronal_view:
            self.coronal_view.index = x
            self.coronal_view.cross_pos = [self.volume.shape[2]-1-z, y]
            self.coronal_view.update_image()
            
        if self.sender() != self.sagittal_view:
            self.sagittal_view.index = y
            self.sagittal_view.cross_pos = [self.volume.shape[2]-1-z, x]
            self.sagittal_view.update_image()
            
        # Update oblique view center
        self.oblique_view.update_center(x, y, z)

    def on_roi_drawn(self, view, bbox):
        # Convert view-specific bbox to global coordinates
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        
        # Update or create global ROI based on view
        if self.global_roi is None:
            # First ROI being drawn
            if view == 'axial':
                # Use current slice +/- 5 slices for initial depth
                curr_z = self.axial_view.index
                z_extent = min(5, self.volume.shape[2]//10)  # or 10% of volume depth
                self.global_roi = (xmin, ymin, max(0, curr_z - z_extent),
                                 xmax, ymax, min(self.volume.shape[2]-1, curr_z + z_extent))
            elif view == 'coronal':
                curr_x = self.coronal_view.index
                x_extent = min(5, self.volume.shape[1]//10)
                self.global_roi = (max(0, curr_x - x_extent), ymin, zmin,
                                 min(self.volume.shape[1]-1, curr_x + x_extent), ymax, zmax)
            else:  # sagittal
                curr_y = self.sagittal_view.index
                y_extent = min(5, self.volume.shape[0]//10)
                self.global_roi = (xmin, max(0, curr_y - y_extent), zmin,
                                 xmax, min(self.volume.shape[0]-1, curr_y + y_extent), zmax)
        else:
            # Update existing ROI based on the view being modified
            x1, y1, z1, x2, y2, z2 = self.global_roi
            if view == 'axial':
                self.global_roi = (xmin, ymin, min(z1, z2),  # preserve z extent
                                 xmax, ymax, max(z1, z2))
            elif view == 'coronal':
                self.global_roi = (min(x1, x2), ymin, zmin,  # preserve x extent
                                 max(x1, x2), ymax, zmax)
            else:  # sagittal
                self.global_roi = (xmin, min(y1, y2), zmin,  # preserve y extent
                                 xmax, max(y1, y2), zmax)
        
        # Update ROI coordinate display
        x1,y1,z1,x2,y2,z2 = self.global_roi
        self.roi_coords_label.setText(
            f"ROI: ({x1},{y1},{z1}) to ({x2},{y2},{z2})")
        
        # Update all views with global ROI
        def update_view_roi(view):
            if not view.image_label.pixmap():
                return
                
            # Calculate view-specific rectangle based on current slice
            if view.orientation == 'axial':
                curr_z = view.index
                if curr_z < min(z1, z2) or curr_z > max(z1, z2):
                    view.roi_pixrect = None
                else:
                    # Show full XY extent at this Z
                    x1_disp, y1_disp = min(x1, x2), min(y1, y2)
                    x2_disp, y2_disp = max(x1, x2), max(y1, y2)
            elif view.orientation == 'coronal':
                curr_x = view.index
                if curr_x < min(x1, x2) or curr_x > max(x1, x2):
                    view.roi_pixrect = None
                else:
                    # Show full YZ extent at this X
                    x1_disp, y1_disp = min(y1, y2), min(z1, z2)
                    x2_disp, y2_disp = max(y1, y2), max(z1, z2)
            else:  # sagittal
                curr_y = view.index
                if curr_y < min(y1, y2) or curr_y > max(y1, y2):
                    view.roi_pixrect = None
                else:
                    # Show full XZ extent at this Y
                    x1_disp, y1_disp = min(x1, x2), min(z1, z2)
                    x2_disp, y2_disp = max(x1, x2), max(z1, z2)
            
            if view.roi_pixrect is not None:
                # Get current slice dimensions
                slice_shape = view.get_slice().shape
                
                # Convert to display coordinates with proper scaling
                pixmap = view.image_label.pixmap()
                scale_x = pixmap.width() / slice_shape[1]
                scale_y = pixmap.height() / slice_shape[0]
                
                # Map coordinates to slice space
                x1_slice = np.clip(x1_disp, 0, slice_shape[1]-1)
                x2_slice = np.clip(x2_disp, 0, slice_shape[1]-1)
                y1_slice = np.clip(y1_disp, 0, slice_shape[0]-1)
                y2_slice = np.clip(y2_disp, 0, slice_shape[0]-1)
                
                # Convert to screen coordinates
                left = min(x1_slice, x2_slice) * scale_x
                top = min(y1_slice, y2_slice) * scale_y
                width = abs(x2_slice - x1_slice) * scale_x
                height = abs(y2_slice - y1_slice) * scale_y
                
                view.roi_pixrect = QtCore.QRect(int(left), int(top), int(width), int(height))
                
                # Store voxel coordinates for this view
                if view.orientation == 'axial':
                    view.roi_voxel_bbox = (x1, x2, y1, y2, z1, z2)
                elif view.orientation == 'coronal':
                    view.roi_voxel_bbox = (x1, x2, y1, y2, z1, z2)
                else:  # sagittal
                    view.roi_voxel_bbox = (x1, x2, y1, y2, z1, z2)
            
            view.update_image()
        
        # Update all views
        update_view_roi(self.axial_view)
        update_view_roi(self.coronal_view)
        update_view_roi(self.sagittal_view)
        
        # Emit current voxel to keep crosshairs synchronized
        self.on_slice_changed(x1, y1, z1)

    def on_roi_cleared(self, view):
        # Clear global ROI
        self.global_roi = None
        self.roi_coords_label.setText("No ROI Selected")
        
        # Clear ROI in all views
        for v in [self.axial_view, self.coronal_view, self.sagittal_view]:
            v.roi_pixrect = None
            v.roi_voxel_bbox = None
            v.set_navigation_limits(None)
            v.update_image()

    def save_roi_volume(self):
        if self.global_roi is None:
            QtWidgets.QMessageBox.warning(self, "No ROI", 
                "Please draw an ROI first before saving.")
            return
            
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save ROI Volume", "", "NIfTI files (*.nii.gz);;All Files (*.*)")
        
        if not filename:
            return
            
        # Ensure .nii.gz extension
        if not filename.endswith('.nii.gz'):
            filename += '.nii.gz'
            
        # Extract ROI volume
        x1, y1, z1, x2, y2, z2 = self.global_roi
        roi_volume = self.volume[
            min(y1,y2):max(y1,y2)+1,
            min(x1,x2):max(x1,x2)+1,
            min(z1,z2):max(z1,z2)+1
        ]
        
        # Calculate new affine for ROI
        new_affine = self.affine.copy()
        # Adjust origin based on ROI start position
        new_affine[:3, 3] = (
            self.affine[:3, :3] @ 
            np.array([min(x1,x2), min(y1,y2), min(z1,z2)]) + 
            self.affine[:3, 3]
        )
        
        # Save as NIfTI
        nii = nib.Nifti1Image(roi_volume, new_affine)
        nib.save(nii, filename)
        
        QtWidgets.QMessageBox.information(self, "Success", 
            f"ROI volume saved successfully to:\n{filename}")

    def save_view(self, orientation):
        sender = self.sender()
        if sender is None:
            return
            
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Volume", "", "NIfTI files (*.nii.gz);;All Files (*.*)")
        
        if not filename:
            return
            
        # Ensure .nii.gz extension
        if not filename.endswith('.nii.gz'):
            filename += '.nii.gz'
            
        # Get ROI if exists
        roi = sender.roi_voxel_bbox
        volume_to_save = self.volume
        affine_to_save = self.affine.copy()
        
        if roi is not None:
            xmin,xmax,ymin,ymax,zmin,zmax = roi
            volume_to_save = self.volume[ymin:ymax+1, xmin:xmax+1, zmin:zmax+1]
            
        # Reorient based on view
        if orientation == 'coronal':
            volume_to_save = np.swapaxes(volume_to_save, 0, 1)
            affine_to_save = np.dot(affine_to_save, np.array([
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]))
        elif orientation == 'sagittal':
            volume_to_save = np.swapaxes(volume_to_save, 0, 2)
            affine_to_save = np.dot(affine_to_save, np.array([
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ]))
            
        # Save as NIfTI
        nii = nib.Nifti1Image(volume_to_save, affine_to_save)
        nib.save(nii, filename)

# ---------------- SingleView ----------------
class SingleView(QtWidgets.QWidget):
    slice_changed = QtCore.pyqtSignal(int, int, int)
    roi_drawn = QtCore.pyqtSignal(str, tuple)
    roi_cleared = QtCore.pyqtSignal(str)
    save_view_requested = QtCore.pyqtSignal(str)

    def __init__(self, volume, affine, orientation, title, color_hex):
        super().__init__()
        self.volume = volume
        self.affine = affine
        self.orientation = orientation
        self.title = title
        self.color_hex = color_hex
        self.H, self.W, self.D = volume.shape

        self.nav_limits = None

        if self.orientation == 'axial':
            self.index = self.D // 2
        elif self.orientation == 'coronal':
            self.index = self.W // 2
        else:
            self.index = self.H // 2

        self.cross_pos = None

        vmin, vmax = float(self.volume.min()), float(self.volume.max())
        self.base_center = (vmax + vmin) / 2.0
        self.base_width = max(1.0, vmax - vmin)
        self.win_center = self.base_center
        self.win_width = self.base_width
        self.zoom = 1.0

        self.playing = False
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._next_slice)

        self.roi_active = False
        self.roi_start = None
        self.roi_end = None
        self.roi_pixrect = None
        self.roi_voxel_bbox = None

        self._build_ui()
        self.update_image(initial=True)

    def _next_slice(self):
        max_idx = self._get_max_index()
        min_idx = self._get_min_index()
        
        self.index = self.index + 1
        if self.index > max_idx:
            self.index = min_idx
            
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(self.index)
        self.slice_slider.blockSignals(False)
        self.update_image()
        self.emit_current_voxel()

    def _get_max_index(self):
        if self.orientation == 'axial':
            base_max = self.D - 1
        elif self.orientation == 'coronal':
            base_max = self.W - 1
        else:
            base_max = self.H - 1
        
        if self.nav_limits is not None:
            return min(base_max, self.nav_limits[1])
        return base_max

    def _get_min_index(self):
        if self.nav_limits is not None:
            return max(0, self.nav_limits[0])
        return 0

    def set_navigation_limits(self, limits):
        self.nav_limits = limits
        if limits is not None:
            self.index = int(np.clip(self.index, limits[0], limits[1]))
            self.slice_slider.setRange(limits[0], limits[1])
        else:
            if self.orientation == 'axial':
                self.slice_slider.setRange(0, self.D - 1)
            elif self.orientation == 'coronal':
                self.slice_slider.setRange(0, self.W - 1)
            else:
                self.slice_slider.setRange(0, self.H - 1)
        
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(self.index)
        self.slice_slider.blockSignals(False)
        self.update_image()

    def _build_ui(self):
        main_v = QtWidgets.QVBoxLayout(self)
        title_lbl = QtWidgets.QLabel(self.title)
        title_lbl.setAlignment(QtCore.Qt.AlignCenter)
        title_lbl.setStyleSheet("font-weight:bold;")
        main_v.addWidget(title_lbl)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(220, 220)
        self.image_label.setStyleSheet("background-color:black;")
        main_v.addWidget(self.image_label, stretch=1)

        bar = QtWidgets.QFrame()
        bar.setFixedHeight(80)
        bar.setStyleSheet(f"background:{self.color_hex}; border-radius:6px;")
        bar_layout = QtWidgets.QVBoxLayout(bar)
        bar_layout.setContentsMargins(6,6,6,6)

        row1 = QtWidgets.QHBoxLayout()
        self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        max_slice = (self.D-1 if self.orientation=='axial' else (self.W-1 if self.orientation=='coronal' else self.H-1))
        self.slice_slider.setRange(0, max_slice)
        self.slice_slider.setValue(self.index)
        self.slice_slider.valueChanged.connect(self.on_slice_slider)
        self.slice_label = QtWidgets.QLabel(f"S:{self.index}/{max_slice}")
        row1.addWidget(QtWidgets.QLabel("Slice"))
        row1.addWidget(self.slice_slider, stretch=1)
        row1.addWidget(self.slice_label)
        bar_layout.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        self.contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.contrast_slider.setRange(1,1000)
        self.contrast_slider.setValue(500)
        self.contrast_slider.valueChanged.connect(self.on_window_sliders)
        row2.addWidget(QtWidgets.QLabel("Contrast"))
        row2.addWidget(self.contrast_slider)

        self.brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brightness_slider.setRange(-1000,1000)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.on_window_sliders)
        row2.addWidget(QtWidgets.QLabel("Brightness"))
        row2.addWidget(self.brightness_slider)
        bar_layout.addLayout(row2)

        main_v.addWidget(bar)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("▶ Play")
        self.btn_pause = QtWidgets.QPushButton("⏸ Pause")
        self.btn_stop = QtWidgets.QPushButton("⏹ Stop")
        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_stop)
        main_v.addLayout(btn_row)

        self.btn_play.clicked.connect(self.cine_play)
        self.btn_pause.clicked.connect(self.cine_pause)
        self.btn_stop.clicked.connect(self.cine_stop)

        self._add_orientation_labels()

    def _add_orientation_labels(self):
        def make_label(text):
            l = QtWidgets.QLabel(text, self.image_label)
            l.setStyleSheet("color:white; background-color: rgba(0,0,0,120); font-weight:bold;")
            l.setFixedSize(24,20)
            l.setAlignment(QtCore.Qt.AlignCenter)
            l.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            l.show()
            return l
        if self.orientation == 'axial':
            top,bottom,left,right = 'A','P','R','L'
        elif self.orientation == 'coronal':
            top,bottom,left,right = 'S','I','R','L'
        else:
            top,bottom,left,right = 'S','I','A','P'
        self.lbl_top = make_label(top)
        self.lbl_bottom = make_label(bottom)
        self.lbl_left = make_label(left)
        self.lbl_right = make_label(right)

        orig_resize = self.image_label.resizeEvent if hasattr(self.image_label, 'resizeEvent') else None
        def resize_event(ev):
            w = self.image_label.width(); h = self.image_label.height()
            self.lbl_top.move(w//2 - 12, 4)
            self.lbl_bottom.move(w//2 - 12, h - 24)
            self.lbl_left.move(4, h//2 - 10)
            self.lbl_right.move(w - 28, h//2 - 10)
            if orig_resize:
                orig_resize(ev)
        self.image_label.resizeEvent = resize_event

    def get_slice(self):
        if self.orientation == 'axial':
            return np.asarray(self.volume[:, :, int(self.index)])
        elif self.orientation == 'coronal':
            return np.rot90(self.volume[:, int(self.index), :])
        else:  # sagittal
            return np.rot90(self.volume[int(self.index), :, :])

    def apply_window(self, img):
        c,w = self.win_center, max(1.0, self.win_width)
        low,high = c - w/2.0, c + w/2.0
        out = np.clip((img - low)/(high - low), 0.0, 1.0)
        return (out * 255.0).astype(np.uint8)

    def draw_crosshair(self, arr):
        arr = arr.copy()
        if self.cross_pos is None:
            self.cross_pos = [arr.shape[0]//2, arr.shape[1]//2]
        r,c = int(np.clip(self.cross_pos[0], 0, arr.shape[0]-1)), int(np.clip(self.cross_pos[1], 0, arr.shape[1]-1))
        arr[r, :] = 255
        arr[:, c] = 255
        return arr

    def update_image(self, initial=False):
        try:
            arr_raw = self.get_slice()
        except Exception as e:
            print("get_slice error:", e)
            return
        arr = self.apply_window(arr_raw)
        arr = self.draw_crosshair(arr)
        im = Image.fromarray(arr)
        if self.zoom != 1.0:
            new_size = (max(1, int(arr.shape[1]*self.zoom)), max(1, int(arr.shape[0]*self.zoom)))
            im = im.resize(new_size, Image.BILINEAR)
        qimg = QtGui.QImage(im.tobytes(), im.width, im.height, im.width, QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(qimg)

        if self.roi_pixrect is not None:
            painter = QtGui.QPainter(pix)
            pen = QtGui.QPen(QtGui.QColor(0,200,255))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(self.roi_pixrect)
            painter.end()

        self.image_label.setPixmap(pix)
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(self.index)
        self.slice_slider.blockSignals(False)
        self.slice_label.setText(f"S:{self.index}/{self.slice_slider.maximum()}")
        if initial:
            self.cross_pos = [arr.shape[0]//2, arr.shape[1]//2]

    def _display_to_arr(self, pt):
        pix = self.image_label.pixmap()
        if pix is None:
            return None
        disp_w, disp_h = pix.width(), pix.height()
        arr = self.get_slice()
        arr_h, arr_w = arr.shape
        x = max(0, min(pt.x(), disp_w-1))
        y = max(0, min(pt.y(), disp_h-1))
        c = int(x / self.zoom)
        r = int(y / self.zoom)
        r = int(np.clip(r, 0, arr_h-1))
        c = int(np.clip(c, 0, arr_w-1))
        return (r, c)

    def _arr_to_voxel(self, r, c):
        if self.orientation == 'axial':
            y = int(r); x = int(c); z = int(self.index)
        elif self.orientation == 'coronal':
            y = int(c); x = int(self.index); z = int((self.D - 1) - r)
        else:
            y = int(self.index); x = int(c); z = int((self.D - 1) - r)
        return (x,y,z)

    def mousePressEvent(self, e):
        rel = self.image_label.mapFromParent(e.pos())
        pix = self.image_label.pixmap()
        if pix is None:
            return super().mousePressEvent(e)
        if not (0 <= rel.x() < pix.width() and 0 <= rel.y() < pix.height()):
            return super().mousePressEvent(e)

        if e.button() == QtCore.Qt.LeftButton:
            self.roi_active = True
            self.roi_start = rel
            self.roi_end = None
            arr_coords = self._display_to_arr(rel)
            if arr_coords:
                r,c = arr_coords
                self.cross_pos = [r,c]
                xv,yv,zv = self._arr_to_voxel(r,c)
                self.slice_changed.emit(int(xv), int(yv), int(zv))
            self.update_image()
        elif e.button() == QtCore.Qt.RightButton:
            if self.roi_pixrect is not None or self.roi_voxel_bbox is not None:
                self.roi_pixrect = None
                self.roi_voxel_bbox = None
                self.roi_active = False
                self.roi_start = None
                self.roi_end = None
                self.update_image()
                self.roi_cleared.emit(self.orientation)
            else:
                self._wl_last = e.pos()

    def mouseMoveEvent(self, e):
        pix = self.image_label.pixmap()
        if pix is None:
            return super().mouseMoveEvent(e)
        if self.roi_active and self.roi_start:
            rel = self.image_label.mapFromParent(e.pos())
            x1,y1 = self.roi_start.x(), self.roi_start.y()
            x2,y2 = rel.x(), rel.y()
            left, top = min(x1,x2), min(y1,y2)
            w = abs(x2-x1); h = abs(y2-y1)
            self.roi_pixrect = QtCore.QRect(left, top, w, h)
            self.roi_end = rel
            self.update_image()
            return
        if e.buttons() & QtCore.Qt.RightButton and hasattr(self, '_wl_last'):
            dx = e.pos().x() - self._wl_last.x()
            dy = e.pos().y() - self._wl_last.y()
            self.win_width = max(1.0, self.win_width + dx * 2.0)
            self.win_center = self.win_center - dy * 2.0
            mult = self.win_width / self.base_width
            val = int(np.clip(mult * 500.0, 1, 1000))
            self.contrast_slider.blockSignals(True)
            self.contrast_slider.setValue(val)
            self.contrast_slider.blockSignals(False)
            offset = int((self.win_center - self.base_center) / (self.base_width / 1000.0))
            offset = int(np.clip(offset, -1000, 1000))
            self.brightness_slider.blockSignals(True)
            self.brightness_slider.setValue(offset)
            self.brightness_slider.blockSignals(False)
            self._wl_last = e.pos()
            self.update_image()
            return
        return super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        pix = self.image_label.pixmap()
        if pix is None:
            return super().mouseReleaseEvent(e)
        if e.button() == QtCore.Qt.LeftButton and self.roi_active and self.roi_start and self.roi_end:
            x1,y1 = self.roi_start.x(), self.roi_start.y()
            x2,y2 = self.roi_end.x(), self.roi_end.y()
            corners = [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]
            vox_pts = []
            for cx,cy in corners:
                arr_coords = self._display_to_arr(QtCore.QPoint(cx,cy))
                if arr_coords is None: continue
                r,c = arr_coords
                voxel = self._arr_to_voxel(r,c)
                if voxel: vox_pts.append(voxel)
            if vox_pts:
                xs = [p[0] for p in vox_pts]; ys = [p[1] for p in vox_pts]; zs = [p[2] for p in vox_pts]
                xmin,xmax = int(min(xs)), int(max(xs))
                ymin,ymax = int(min(ys)), int(max(ys))
                zmin,zmax = int(min(zs)), int(max(zs))
                self.roi_voxel_bbox = (xmin,xmax,ymin,ymax,zmin,zmax)
                self.roi_drawn.emit(self.orientation, self.roi_voxel_bbox)
            self.roi_active = False
            self.update_image()
            return
        if e.button() == QtCore.Qt.RightButton and hasattr(self, '_wl_last'):
            delattr(self, '_wl_last')
            return
        return super().mouseReleaseEvent(e)

    def on_slice_slider(self, val):
        self.index = int(val)
        self.update_image()
        self.emit_current_voxel()

    def on_window_sliders(self, _v):
        mult = self.contrast_slider.value() / 500.0
        self.win_width = max(1.0, self.base_width * mult)
        offset = self.brightness_slider.value()
        self.win_center = self.base_center + offset * (self.base_width / 1000.0)
        self.update_image()

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        mods = e.modifiers()
        if mods == QtCore.Qt.ControlModifier:
            if delta > 0: self.zoom *= 1.1
            else: self.zoom /= 1.1
            self.zoom = float(np.clip(self.zoom, 0.1, 10.0))
            self.update_image()
        else:
            max_idx = self._get_max_index()
            min_idx = self._get_min_index()
            if delta > 0:
                self.index = int(np.clip(self.index+1, min_idx, max_idx))
            else:
                self.index = int(np.clip(self.index-1, min_idx, max_idx))
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(self.index)
            self.slice_slider.blockSignals(False)
            self.update_image()
            self.emit_current_voxel()

    def emit_current_voxel(self):
        arr = self.get_slice()
        arr_h, arr_w = arr.shape
        r = int(np.clip(self.cross_pos[0] if self.cross_pos is not None else arr_h//2, 0, arr_h-1))
        c = int(np.clip(self.cross_pos[1] if self.cross_pos is not None else arr_w//2, 0, arr_w-1))
        x,y,z = self._arr_to_voxel(r,c)
        self.slice_changed.emit(int(x), int(y), int(z))

    def cine_play(self):
        if not self.playing:
            self.playing = True
            self.timer.start(80)

    def cine_pause(self):
        if self.playing:
            self.timer.stop()
            self.playing = False

    def cine_stop(self):
        if self.playing:
            self.timer.stop()
            self.playing = False
        if self.orientation == 'axial':
            self.index = self.D // 2
        elif self.orientation == 'coronal':
            self.index = self.W // 2
        else:
            self.index = self.H // 2
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(self.index)
        self.slice_slider.blockSignals(False)
        self.update_image()
        self.emit_current_voxel()

# ---------------- ObliqueView ----------------
class ObliqueView(QtWidgets.QWidget):
    def __init__(self, volume, affine):
        super().__init__()
        self.volume = volume
        self.affine = affine
        self.H, self.W, self.D = volume.shape
        
        self.center = np.array([self.W//2, self.H//2, self.D//2], dtype=float)
        self.angles = [0.0, 0.0, 0.0]
        self.slice_size = 256
        
        vmin, vmax = float(self.volume.min()), float(self.volume.max())
        self.base_center = (vmax + vmin) / 2.0
        self.base_width = max(1.0, vmax - vmin)
        self.win_center = self.base_center
        self.win_width = self.base_width
        self.zoom = 1.0
        
        self._build_ui()
        self.update_image()

    def _build_ui(self):
        main_v = QtWidgets.QVBoxLayout(self)
        title_lbl = QtWidgets.QLabel("Oblique View")
        title_lbl.setAlignment(QtCore.Qt.AlignCenter)
        title_lbl.setStyleSheet("font-weight:bold; font-size:14px;")
        main_v.addWidget(title_lbl)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(220, 220)
        self.image_label.setStyleSheet("background-color:black;")
        main_v.addWidget(self.image_label, stretch=1)

        control_frame = QtWidgets.QFrame()
        control_frame.setStyleSheet("background:#9b59b6; border-radius:6px; padding:10px;")
        control_layout = QtWidgets.QVBoxLayout(control_frame)
        
        rx_layout = QtWidgets.QHBoxLayout()
        rx_layout.addWidget(QtWidgets.QLabel("Rotate X:"))
        self.rx_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rx_slider.setRange(-180, 180)
        self.rx_slider.setValue(0)
        self.rx_slider.valueChanged.connect(self.on_rotation_changed)
        self.rx_label = QtWidgets.QLabel("0°")
        self.rx_label.setFixedWidth(40)
        rx_layout.addWidget(self.rx_slider)
        rx_layout.addWidget(self.rx_label)
        control_layout.addLayout(rx_layout)
        
        ry_layout = QtWidgets.QHBoxLayout()
        ry_layout.addWidget(QtWidgets.QLabel("Rotate Y:"))
        self.ry_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ry_slider.setRange(-180, 180)
        self.ry_slider.setValue(0)
        self.ry_slider.valueChanged.connect(self.on_rotation_changed)
        self.ry_label = QtWidgets.QLabel("0°")
        self.ry_label.setFixedWidth(40)
        ry_layout.addWidget(self.ry_slider)
        ry_layout.addWidget(self.ry_label)
        control_layout.addLayout(ry_layout)
        
        rz_layout = QtWidgets.QHBoxLayout()
        rz_layout.addWidget(QtWidgets.QLabel("Rotate Z:"))
        self.rz_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rz_slider.setRange(-180, 180)
        self.rz_slider.setValue(0)
        self.rz_slider.valueChanged.connect(self.on_rotation_changed)
        self.rz_label = QtWidgets.QLabel("0°")
        self.rz_label.setFixedWidth(40)
        rz_layout.addWidget(self.rz_slider)
        rz_layout.addWidget(self.rz_label)
        control_layout.addLayout(rz_layout)
        
        wl_layout = QtWidgets.QHBoxLayout()
        self.contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.contrast_slider.setRange(1, 1000)
        self.contrast_slider.setValue(500)
        self.contrast_slider.valueChanged.connect(self.on_window_changed)
        wl_layout.addWidget(QtWidgets.QLabel("Contrast:"))
        wl_layout.addWidget(self.contrast_slider)
        
        self.brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brightness_slider.setRange(-1000, 1000)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.on_window_changed)
        wl_layout.addWidget(QtWidgets.QLabel("Brightness:"))
        wl_layout.addWidget(self.brightness_slider)
        control_layout.addLayout(wl_layout)
        
        reset_btn = QtWidgets.QPushButton("Reset Orientation")
        reset_btn.clicked.connect(self.reset_orientation)
        control_layout.addWidget(reset_btn)
        
        main_v.addWidget(control_frame)

    def on_rotation_changed(self, _):
        self.angles = [
            self.rx_slider.value(),
            self.ry_slider.value(),
            self.rz_slider.value()
        ]
        self.rx_label.setText(f"{self.angles[0]}°")
        self.ry_label.setText(f"{self.angles[1]}°")
        self.rz_label.setText(f"{self.angles[2]}°")
        self.update_image()

    def reset_orientation(self):
        self.angles = [0.0, 0.0, 0.0]
        self.rx_slider.setValue(0)
        self.ry_slider.setValue(0)
        self.rz_slider.setValue(0)
        self.update_image()

    def update_center(self, x, y, z):
        self.center = np.array([x, y, z], dtype=float)
        self.update_image()

    def on_window_changed(self, _):
        mult = self.contrast_slider.value() / 500.0
        self.win_width = max(1.0, self.base_width * mult)
        offset = self.brightness_slider.value()
        self.win_center = self.base_center + offset * (self.base_width / 1000.0)
        self.update_image()

    def wheelEvent(self, e):
        if e.modifiers() == QtCore.Qt.ControlModifier:
            delta = e.angleDelta().y()
            if delta > 0:
                self.zoom *= 1.1
            else:
                self.zoom /= 1.1
            self.zoom = float(np.clip(self.zoom, 0.1, 10.0))
            self.update_image()

    def update_image(self):
        rot_matrix = rotation_matrix_from_angles(self.angles)
        normal = rot_matrix @ np.array([0, 0, 1])
        up_vector = rot_matrix @ np.array([0, -1, 0])
        
        slice_data = extract_oblique_slice(self.volume, self.center, normal, up_vector, (self.slice_size, self.slice_size))
        
        c, w = self.win_center, max(1.0, self.win_width)
        low, high = c - w/2.0, c + w/2.0
        arr = np.clip((slice_data - low)/(high - low), 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
        
        im = Image.fromarray(arr)
        if self.zoom != 1.0:
            new_size = (max(1, int(arr.shape[1]*self.zoom)), max(1, int(arr.shape[0]*self.zoom)))
            im = im.resize(new_size, Image.BILINEAR)
            
        qimg = QtGui.QImage(im.tobytes(), im.width, im.height, im.width, QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Ask user to select between DICOM and NIfTI
    msg_box = QtWidgets.QMessageBox()
    msg_box.setIcon(QtWidgets.QMessageBox.Question)
    msg_box.setText("Select image format to load:")
    msg_box.setWindowTitle("Select Format")
    nifti_button = msg_box.addButton("NIfTI", QtWidgets.QMessageBox.ActionRole)
    dicom_button = msg_box.addButton("DICOM", QtWidgets.QMessageBox.ActionRole)
    cancel_button = msg_box.addButton(QtWidgets.QMessageBox.Cancel)
    
    msg_box.exec_()
    clicked_button = msg_box.clickedButton()
    
    file_path = None
    if clicked_button == nifti_button:
        file_path = get_nifti_file()
    elif clicked_button == dicom_button:
        file_path = get_dicom_folder()
    else:
        sys.exit()
        
    if file_path is None:
        sys.exit()
    
    try:
        main_window = MainWindow(file_path)
        main_window.show()
        sys.exit(app.exec_())
    except Exception as e:
        error_msg = QtWidgets.QMessageBox()
        error_msg.setIcon(QtWidgets.QMessageBox.Critical)
        error_msg.setText(f"Error loading image:\n{str(e)}")
        error_msg.setWindowTitle("Error")
        error_msg.exec_()
        sys.exit(1)
        