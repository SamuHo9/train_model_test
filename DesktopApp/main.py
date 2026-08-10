import sys
import os
import vtk
import webbrowser
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, QFileDialog,
                             QSplitter, QToolBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QComboBox, QLineEdit, QMessageBox)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hippocampal Shape Analysis Pipeline (Slicer-style)")
        self.resize(1200, 800)

        self.create_menus_and_toolbar()

        # Main central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # QSplitter to divide Left (Toolbox) and Right (Viewer)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ----------------------------------------------------
        # 1. Left Panel (Toolbox)
        # ----------------------------------------------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # --- Logo / Header ---
        logo_label = QLabel("Shape Analysis Toolbox\nHippocampal Pipeline")
        font = logo_label.font()
        font.setPointSize(14)
        font.setBold(True)
        logo_label.setFont(font)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet("color: #2c3e50; padding: 10px; background-color: #ecf0f1; border-radius: 5px;")
        left_layout.addWidget(logo_label)

        # --- QToolBox (Collapsible Panels) ---
        self.toolbox = QToolBox()
        left_layout.addWidget(self.toolbox)

        # Panel 1: Help & Acknowledgement
        help_widget = QWidget()
        help_layout = QVBoxLayout(help_widget)
        help_label = QLabel("This project is for Hippocampal Shape Analysis.\nUse the tools below to run the ICP -> SPHARM -> PLS-DA pipeline.")
        help_label.setWordWrap(True)
        help_layout.addWidget(help_label)
        help_layout.addStretch()
        self.toolbox.addItem(help_widget, "Help & Acknowledgement")

        # Panel 2: Import Data Properties
        import_widget = QWidget()
        import_layout = QVBoxLayout(import_widget)
        
        btn_layout = QHBoxLayout()
        btn_import_dir = QPushButton("Import from directory")
        btn_import_csv = QPushButton("Import from CSV")
        btn_layout.addWidget(btn_import_dir)
        btn_layout.addWidget(btn_import_csv)
        import_layout.addLayout(btn_layout)

        dir_select_btn = QPushButton("📁 Choose Data Directory")
        dir_select_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 5px;")
        dir_select_btn.clicked.connect(self.select_directory)
        import_layout.addWidget(dir_select_btn)

        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Folder:"))
        self.folder_input = QLineEdit()
        self.folder_input.setReadOnly(True)
        folder_layout.addWidget(self.folder_input)
        import_layout.addLayout(folder_layout)
        
        import_action_btn = QPushButton("Import")
        import_action_btn.setStyleSheet("background-color: #2980b9; color: white; padding: 5px;")
        import_layout.addWidget(import_action_btn)
        
        import_layout.addStretch()
        self.toolbox.addItem(import_widget, "Import Data Properties")

        # Panel 3: Imported Subjects
        subjects_widget = QWidget()
        subjects_layout = QVBoxLayout(subjects_widget)
        self.subjects_table = QTableWidget(0, 2)
        self.subjects_table.setHorizontalHeaderLabels(["Subject name", "Consistency"])
        self.subjects_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        subjects_layout.addWidget(self.subjects_table)
        
        self.add_mock_subjects()
        self.toolbox.addItem(subjects_widget, "Imported Subjects")

        # Panel 4: Analysis & Logging
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        
        self.run_icp_btn = QPushButton("Run ICP Registration")
        self.run_icp_btn.clicked.connect(lambda: self.log(">>> Running ICP Registration..."))
        self.run_spharm_btn = QPushButton("Run SPHARM Processing")
        self.run_spharm_btn.clicked.connect(lambda: self.log(">>> Running SPHARM Processing..."))
        self.run_plsda_btn = QPushButton("Run PLS-DA Analysis")
        self.run_plsda_btn.clicked.connect(lambda: self.log(">>> Running PLS-DA Analysis..."))
        
        analysis_layout.addWidget(self.run_icp_btn)
        analysis_layout.addWidget(self.run_spharm_btn)
        analysis_layout.addWidget(self.run_plsda_btn)
        
        analysis_layout.addWidget(QLabel("Console Output:"))
        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        analysis_layout.addWidget(self.log_window)
        self.toolbox.addItem(analysis_widget, "Analysis & Logging")

        # ----------------------------------------------------
        # 2. Right Panel (3D Viewer)
        # ----------------------------------------------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # VTK Widget
        self.vtk_widget = QVTKRenderWindowInteractor(right_panel)
        right_layout.addWidget(self.vtk_widget)

        # Setup VTK Rendering pipeline
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        self.add_vtk_placeholder()

        # Add left and right to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

        # Set initial splitter sizes (approx 30% left, 70% right)
        splitter.setSizes([350, 850])

        self.log("SlicerSALT-style UI initialized successfully.")
        
        # Check for SlicerSALT dependency
        self.check_slicer_salt()

    def create_menus_and_toolbar(self):
        # Menu Bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        edit_menu = menubar.addMenu("Edit")
        view_menu = menubar.addMenu("View")

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Tool Bar
        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setMovable(False)

        # Add mock buttons to the toolbar to match the screenshot
        save_action = QAction("💾 Save", self)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()

        # Add Module Selector to Toolbar
        toolbar.addWidget(QLabel("  Modules: "))
        self.module_combo = QComboBox()
        self.module_combo.addItems(["Data Importer", "ICP Registration", "SPHARM Processing", "PLS-DA Analysis", "Feature Extraction"])
        self.module_combo.setMinimumWidth(200)
        toolbar.addWidget(self.module_combo)

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if directory:
            self.folder_input.setText(directory)
            self.log(f"Selected directory: {directory}")

    def check_slicer_salt(self):
        import glob
        # Look for SlicerSALT inside Program Files based on the start_salt_batch.bat script
        slicer_paths = glob.glob(r"C:\Program Files\SlicerSALT*\SlicerSALT.exe")
        
        if not slicer_paths:
            QMessageBox.critical(self, "SlicerSALT Required", 
                                "SlicerSALT could not be found in the default installation directory (C:\\Program Files\\SlicerSALT*).\n\n"
                                "This program requires SlicerSALT to function. The application will now close and open the download page.")
            webbrowser.open("https://salt.slicer.org/")
            sys.exit(1)
        else:
            self.log(f"SUCCESS: SlicerSALT detected at {slicer_paths[0]}")

    def add_mock_subjects(self):
        # Adding some mock data to visually match the screenshot you provided
        mock_data = [
            ("001_tp3CranialReg.nrrd", "OK"),
            ("001_tp1CranialReg.nrrd", "# Inconsistencies: 1"),
            ("001_tp2CranialReg.nrrd", "# Inconsistencies: 1")
        ]
        self.subjects_table.setRowCount(len(mock_data))
        for row, (name, cons) in enumerate(mock_data):
            self.subjects_table.setItem(row, 0, QTableWidgetItem(name))
            self.subjects_table.setItem(row, 1, QTableWidgetItem(cons))

    def add_vtk_placeholder(self):
        """Adds a 3D placeholder (sphere) to the VTK viewer to mimic the mesh view."""
        self.renderer.SetBackground(0.8, 0.8, 0.9) # Light purple/blue background like Slicer
        
        # Create a sphere to simulate a hippocampal mesh
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(10.0)
        sphere_source.SetPhiResolution(50)
        sphere_source.SetThetaResolution(50)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere_source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # Flesh/Bone color
        actor.GetProperty().SetColor(0.8, 0.6, 0.5)

        self.renderer.AddActor(actor)
        
        # Reset camera
        self.renderer.ResetCamera()
        
        # Start interactor
        self.interactor.Initialize()

    def log(self, message):
        self.log_window.append(message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # ----------------------------------------------------
    # Splash Screen (Slicer/SlicerSALT style startup)
    # ----------------------------------------------------
    from PyQt6.QtGui import QPixmap, QColor, QFont, QPainter, QPen
    from PyQt6.QtWidgets import QSplashScreen
    import time
    
    # Create a dynamic pixmap for the splash screen
    pixmap = QPixmap(600, 350)
    pixmap.fill(QColor("#2c3e50")) # Dark blue Slicer-style background
    
    # Draw a simple logo/title directly on the splash image
    painter = QPainter(pixmap)
    painter.setPen(QPen(QColor("white")))
    painter.setFont(QFont("Arial", 20, QFont.Weight.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Shape Analysis Toolbox\nInitializing...")
    painter.end()

    splash = QSplashScreen(pixmap)
    splash.setFont(QFont("Arial", 11, QFont.Weight.Bold))
    splash.show()
    
    # Simulate loading steps with visual feedback
    splash.showMessage("Starting Hippocampal Shape Analysis Pipeline...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    app.processEvents()
    time.sleep(0.6)
    
    splash.showMessage("Loading VTK Rendering Engine...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    app.processEvents()
    time.sleep(0.6)
    
    splash.showMessage("Checking System Requirements & SlicerSALT...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    app.processEvents()
    time.sleep(0.6)
    
    splash.showMessage("Initializing User Interface Modules...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    app.processEvents()
    time.sleep(0.4)
    
    # Initialize the heavy main window
    window = MainWindow()
    
    # Close splash and show main window
    splash.finish(window)
    window.show()
    sys.exit(app.exec())
