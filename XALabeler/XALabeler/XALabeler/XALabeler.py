# Install
import slicer
try:
    import sklearn
except:
    slicer.util.pip_install("scikit-learn")
    import sklearn

# Imports
import logging
import os
import sys
import vtk
import time
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import SegmentStatistics
from settings.settings import Settings
from settings.settingsPath import settingsPath
from ALAction.ALAction import ALAction
import qt
import json
import numpy as np 
import SimpleITK as sitk
from glob import glob
from sklearn.manifold import TSNE

def scale_between_quantiles(x, q_low=0.01, q_high=0.99, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    lo = np.quantile(x, q_low)
    hi = np.quantile(x, q_high)
    x_scaled = (x - lo) / (hi - lo + eps)
    x_scaled = np.clip(x_scaled, 0, 1)
    return x_scaled
              
def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            print('"{old_string}" not found in {filename}.'.format(**locals()))
            return

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        print('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
        s = s.replace(old_string, new_string)
        f.write(s)

def makeTableFromXY(name, xs, ys):
    tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)
    t = tableNode.GetTable()

    xArr = vtk.vtkFloatArray(); xArr.SetName("x")
    yArr = vtk.vtkFloatArray(); yArr.SetName("y")
    t.AddColumn(xArr); t.AddColumn(yArr)

    n = len(xs)
    t.SetNumberOfRows(n)
    for i in range(n):
        t.SetValue(i, 0, float(xs[i]))
        t.SetValue(i, 1, float(ys[i]))
    return tableNode

def ensureAtLeastTwoRows(tableNode):
    t = tableNode.GetTable()
    if t.GetNumberOfRows() == 1:
        t.InsertNextBlankRow()  # creates row index 1
        for c in range(t.GetNumberOfColumns()):
            t.SetValue(1, c, t.GetValue(0, c))  # duplicate row 0 -> row 1
        tableNode.Modified()

def addScatterSeries(chartNode, tableNode, seriesName, rgb=None, markerSize=10):
    if tableNode.GetTable().GetNumberOfRows() < 1:
        return None

    s = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", seriesName)

    wasMod = s.StartModify()
    try:
        # Set plot properties FIRST
        s.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        s.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleNone)  # <-- use enum, not 0
        s.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleCircle)
        s.SetMarkerSize(markerSize)
        s.SetXColumnName("x")
        s.SetYColumnName("y")

        if rgb is None:
            s.SetUniqueColor()
        else:
            s.SetColor(float(rgb[0]), float(rgb[1]), float(rgb[2]))

        # Attach table LAST (avoids default-line render on attach)
        ensureAtLeastTwoRows(tableNode)
        s.SetAndObserveTableNodeID(tableNode.GetID())
    finally:
        s.EndModify(wasMod)

    chartNode.AddAndObservePlotSeriesNodeID(s.GetID())
    return s

def setScatterColor(series, color=[100,100,100]):

    wasMod = series.StartModify()
    try:
        if color is None:
            series.SetUniqueColor()
        else:
            series.SetColor(float(color[0]), float(color[1]), float(color[2]))
    finally:
        series.EndModify(wasMod)

#
# XALabeler
#

class XALabeler(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "XALabeler"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["XAL"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#XALabeler">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # XALabeler1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='XALabeler',
        sampleName='XALabeler1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'XALabeler1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='XALabeler1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='XALabeler1'
    )

    # XALabeler2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='XALabeler',
        sampleName='XALabeler2',
        thumbnailFileName=os.path.join(iconsPath, 'XALabeler2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='XALabeler2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='XALabeler2'
    )


#
# XALabelerWidget
#

class XALabelerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.actionlist = []
        self.actionIdx = -1
        self.volumeNode = None
        self.pseudoNode = None
        self.maskNode = None
        self.ucNode = None
        self.roiNode = []
        self.mip = False
        
    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/XALabeler.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = XALabelerLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Setup SegmentEditorWidget
        self.ui.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        self.segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.ui.segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)
        self.ui.segmentEditorWidget.setEffectNameOrder(["Paint", "Erase"])
        self.ui.segmentEditorWidget.unorderedEffectsVisible = False
        self.ui.segmentEditorWidget.setActiveEffectByName("Paint")
        self.ui.segmentEditorWidget.show()

        # Update layoutManager
        self.layoutManager = slicer.app.layoutManager()
        self.layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
        self.red = self.layoutManager.sliceWidget('Red')
        self.redLogic = self.red.sliceLogic()
        
        # Read settings
        self.settings = Settings()
        #fip_settings = os.path.dirname(os.path.dirname(
        #    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + '/XALabeler/data_manual/settings_XALabeler.json'
        fip_settings = settingsPath
        if not os.path.exists(fip_settings):
            self.settings.writeSettings(fip_settings)
        self.settings.readSettings(fip_settings)

        # Update label
        self.ui.label.setText('Welcome to XALabeler!')
        
        # Buttons
        self.ui.startSelectionButton.connect('clicked(bool)', self.onStartSelectionButton)
        self.ui.startSelectionButton.setEnabled(True)
        self.ui.alphaBox.setEnabled(False)
        self.ui.betaBox.setEnabled(False)
        self.ui.gammaBox.setEnabled(False)
        self.ui.deltaBox.setEnabled(False)
        self.ui.thetaBox.setEnabled(False)
        self.ui.strataBox.setEnabled(False)

        #self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.ui.startAnnotationButton.connect('clicked(bool)', self.onStartAnnotationButton)
        self.ui.nextButton.connect('clicked(bool)', self.onNextButton)
        self.ui.backButton.connect('clicked(bool)', self.onBackButton)
        self.ui.stopButton.connect('clicked(bool)', self.onStopButton)

        self.selectPseudo = False
        self.ui.pseudoCheckBox.toggled.connect(self.onPseudoCheckBox)
        self.ui.pseudoCheckBox.setChecked(self.selectPseudo)

        self.ui.startAnnotationButton.setEnabled(False)
        self.ui.nextButton.setEnabled(False)
        self.ui.backButton.setEnabled(False)
        self.ui.stopButton.setEnabled(False)
        self.ui.pseudoCheckBox.setEnabled(False)

        # Checkbox
        self.ui.selectionCheckBox.toggled.connect(self.onSelectionCheckBox)
        self.selection=True
        self.ui.selectionCheckBox.setChecked(not self.selection)
        # Checkbox
        self.ui.tsneCheckBox.toggled.connect(self.onTsneCheckBox)
        #self.tsne_update = False
        self.ui.tsneCheckBox.setChecked(False)

        # Parameters
        self.alpha = 1.0
        self.ui.alphaBox.setValue(self.alpha)
        self.ui.alphaBox.valueChanged.connect(self.updateAlpha)
        self.beta = 1.0
        self.ui.betaBox.setValue(self.beta)
        self.ui.betaBox.valueChanged.connect(self.updateBeta)
        self.gamma = 1.0
        self.ui.gammaBox.setValue(self.gamma)
        self.ui.gammaBox.valueChanged.connect(self.updateGamma)
        self.delta = 0.0
        self.ui.deltaBox.setValue(self.delta)
        self.ui.deltaBox.valueChanged.connect(self.updateDelta)
        self.theta = 0.0
        self.ui.thetaBox.setValue(self.theta)
        self.ui.thetaBox.valueChanged.connect(self.updateTheta)
        self.strata = 'NONE'
        #self.ui.strataBox.addItems(["NONE", "CLASS"])
        #self.ui.strataBox.addItems(["NONE"])
        self.ui.strataBox.setCurrentText(self.strata)
        self.ui.strataBox.currentTextChanged.connect(self.updateStrata)

        # Textbox
        self.ui.textBatch.setPlainText("Welcome to XALabeler!")
        
        # Progress bar
        self.ui.progressBar.setValue(0)
        
        # Extract color labels
        #print('fip_colors123', self.settings['fip_colors'])

        colorNode = slicer.util.loadColorTable(self.settings['fip_colors'])
        for i in range(colorNode.GetNumberOfColors()):
            cname = colorNode.GetColorName(i)
            colorNode.SetColorName(i, cname.replace(' ', '_'))
        self.colorLabels = [colorNode.GetColorName(i) for i in range(colorNode.GetNumberOfColors())]
        
        # Add crosshair node
        self.crosshairNode=slicer.util.getNode("Crosshair")

        # Disable elements
        if not self.settings['clustering']:
            self.ui.alphaBox.setEnabled(False)
            self.ui.betaBox.setEnabled(False)
            self.ui.deltaBox.setEnabled(False)
            self.ui.thetaBox.setEnabled(False)
        else:
            self.ui.progressBar.setEnabled(False)
            #self.ui.horizontalSlider.setEnabled(False)
            


        # Add shortcuts
        shortcuts = [
          ('n', lambda: self.onNextButton()),
          ('b', lambda: self.onBackButton()),
          ('q', lambda: self.onStopButton()),
          ('e', lambda: self.onToggleUncertainty()),
          ('t', lambda: self.onToggleVisibility()),
          ('r', lambda: self.onToggleROIVisibility()),
          ('u', lambda: self.onToggleThreshold()),
          ('z', lambda: self.onSelectLabel()),
          ('s', lambda: self.onSelectPositive()),
          ('a', lambda: self.onSelectNegative()),
          ('p', lambda: self.onTogglePseudo()),
          ('1', lambda: self.onNumberButton(1)),
          ('2', lambda: self.onNumberButton(2)),
          ('3', lambda: self.onNumberButton(3)),
          ('4', lambda: self.onNumberButton(4)),
          ('5', lambda: self.onNumberButton(5)),
          ('6', lambda: self.onNumberButton(6)),
          ('7', lambda: self.onNumberButton(7)),
          ('8', lambda: self.onNumberButton(8)),
          ('9', lambda: self.onNumberButton(9)),
          ('Ctrl+,', lambda: slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)),
          ('m', lambda: self.onMIP())
          ]
        
        for (shortcutKey, callback) in shortcuts:
            shortcut = qt.QShortcut(slicer.util.mainWindow())
            shortcut.setKey(qt.QKeySequence(shortcutKey))
            shortcut.connect( 'activated()', callback)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
        self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
        self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def onStartAnnotationButton(self):
        """
        Run processing when user clicks "Start" button.
        """
        #self.ui.startButton.setEnabled(False)
        self.ui.startAnnotationButton.setEnabled(False)
        self.ui.nextButton.setEnabled(True)
        self.ui.backButton.setEnabled(True)
        self.ui.stopButton.setEnabled(True)
        self.ui.stopButton.setEnabled(True)
        self.ui.pseudoCheckBox.setEnabled(True)

        # Init actionIdx
        self.actionIdx = len(self.actionlist)
        for i, a in enumerate(self.actionlist):
            if a.info['label']['batch'] and a.status == 'open':
                self.actionIdx = i
                break

        # Init actionIdxSave
        self.actionIdxSave=[]

        # Load first slice
        if len(self.actionlist) > 0 and self.actionIdx<len(self.actionlist):
            self.process()

            
    def onStartSelectionButton(self):
        """
        Run processing when user clicks "Start" button.
        """
        #print('onStartSelectionButton1234')
        self.ui.startSelectionButton.setEnabled(False)
        self.ui.alphaBox.setEnabled(True)
        self.ui.betaBox.setEnabled(True)
        self.ui.gammaBox.setEnabled(True)
        self.ui.deltaBox.setEnabled(True)
        self.ui.thetaBox.setEnabled(True)
        self.ui.strataBox.setEnabled(True)

        # Load actionlist
        self.actionlist = ALAction.load(self.settings['fip_actionlist'])

        # Update strata
        labels = [lab[1] for lab in self.actionlist[0].label]
        #print('labels123', labels)

        thr_fg = 0.00001
        for i, a in enumerate(self.actionlist):
            #if 'strata_info' not in a.info:
            a.info['strata_info'] = []
            for i,lab in enumerate(labels[0:-2]):
                if 'fg' in a.info:
                    if a.info['fg'][i]>thr_fg:
                        a.info['strata_info'].append(lab)
            # Add mri/ct info
            if 'amos_05' in a.imagename:
                a.info['strata_info'].append('mri')
            else:
                a.info['strata_info'].append('ct')

        # Check if annotation process was already started
        annotated = False
        for a in self.actionlist:
            #print('a.status123', a.status)
            if a.status=='solved':
                annotated = True
                break
        if annotated:
            #print('annotated123')
            self.ui.label.setText('Annotated samples found! Switching to annotation stage!')
            self.selection = True
            self.ui.selectionCheckBox.setChecked(self.selection)
            return


        # Reset action info
        for a in self.actionlist:
            a.info['label'] = dict()
            a.info['label']['positive'] = False
            a.info['label']['negative'] = False
            a.info['label']['batch'] = False

        # Init actionIdx
        self.actionIdx = len(self.actionlist)
        for i, a in enumerate(self.actionlist):
            if a.status == 'open':
                self.actionIdx = i
                #print('self.actionIdx123', self.actionIdx, a.id, a.imagename)
                break

        # Update instructions
        if self.settings['classification']:
            self.ui.label.setText('Classiefie the image 1-positive, 2-negative.')
        else:
            self.ui.label.setText('Segment the image.')
            
        # Init actionIdxSave
        self.actionIdxSave=[]

        # Update strataBox
        strata_info = []
        for i, a in enumerate(self.actionlist):
            if 'strata_info' in a.info:
                strata_info = strata_info + a.info['strata_info']
        #strata_info = ['NONE'] + list(set(strata_info))
        strata_info = list(set(strata_info))
        strata_info.sort()
        strata_info = ['NONE'] + strata_info
        self.strata_info = strata_info
        self.strata_weight = np.zeros(len(strata_info))

        # Create scatter plot
        if self.settings['clustering']:

            # Cluster using similarity matrix
            fp_manual = os.path.dirname(self.settings['fip_actionlist'])
            fip_grad = os.path.join(fp_manual, "GRADMat.npy")
            if not os.path.isfile(fip_grad):
                Mgrad = np.random.rand(len(self.actionlist), len(self.actionlist))
                Dtsne = Mgrad
                np.fill_diagonal(Dtsne, 0.0)
            else:
                Mgrad = np.load(fip_grad)
                Dtsne = 1.0 - Mgrad
                Dtsne = np.clip(Dtsne, 0.0, 2.0)   # cosine distance range is [0, 2]
                np.fill_diagonal(Dtsne, 0.0)
            self.Dtsne = Dtsne
            tsne = TSNE(random_state=123,n_components=2,verbose=0,perplexity=40,max_iter=300).fit_transform(Dtsne)
            self.embID=[i for i in range(tsne.shape[0])]
            embArray = tsne

            
            if 'uc' in self.actionlist[0].info:
                uc=[]
                for i, action in enumerate(self.actionlist):
                    uc.append(action.info['uc'])
                uc = np.array(uc)
                uc = scale_between_quantiles(uc)
            else:
                uc=[]
                for i, action in enumerate(self.actionlist):
                    uc.append(0.5)
            
            for i, action in enumerate(self.actionlist):
                action.info['color'] = ( int(1+uc[i]*254), int(1+uc[i]*254), int(1+uc[i]*254) )
            #print('uc123', uc)


            plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", "scatter")
            self.plotChartNode = plotChartNode
            self.scatterSeries = []
            for i in range(len(self.actionlist)):
            #for i in range(embArray.shape[0]):
                tNode = makeTableFromXY(f"table_{i}", [tsne[i,0]], [tsne[i,1]])
                rgb = self.actionlist[i].info['color']
                s = addScatterSeries(plotChartNode, tNode, f"series_{i}", rgb=rgb)
                self.scatterSeries.append(s)
            slicer.modules.plots.logic().ShowChartInLayout(plotChartNode)

            # Connect the signal with a slot ''onDataSelected''
            layoutManager = slicer.app.layoutManager()
            self.plotWidget = layoutManager.plotWidget(0)
            self.plotView = self.plotWidget.plotView()

            self.currentSeriesID = None
            self.plotView.connect("dataSelected(vtkStringArray*, vtkCollection*)", self.onScatterSelected)

              # Update scaater plot
            self.updateBatch()

        self.ui.strataBox.addItems(strata_info)
        self.ui.strataBox.setCurrentText(self.strata)

        # Update thetaBox
        idx = self.ui.strataBox.currentIndex
        self.theta = self.strata_weight[idx]

        # Load first slice
        if len(self.actionlist) > 0 and self.actionIdx<len(self.actionlist):
            self.process()

    def onSelectionCheckBox(self, checked):

        self.ui.startSelectionButton.setEnabled(False)
        self.ui.alphaBox.setEnabled(False)
        self.ui.betaBox.setEnabled(False)
        self.ui.gammaBox.setEnabled(False)
        self.ui.deltaBox.setEnabled(False)
        self.ui.thetaBox.setEnabled(False)
        self.ui.strataBox.setEnabled(False)

        self.ui.startAnnotationButton.setEnabled(True)
       
        self.selection=False

    def onTsneCheckBox(self, checked):
        if self.ui.tsneCheckBox.isChecked():
            tsne_L = TSNE(random_state=123,n_components=2,verbose=0,perplexity=40,max_iter=300).fit_transform(self.Ldpp)
        else:
            tsne_L = TSNE(random_state=123,n_components=2,verbose=0,perplexity=40,max_iter=300).fit_transform(self.Dtsne)

        scene = slicer.mrmlScene
        plotLogic = slicer.modules.plots.logic()
        chartNode = self.plotChartNode
        scene.StartState(scene.BatchProcessState)
        mChart = chartNode.StartModify() if chartNode else None

        tableNodes = [s.GetTableNode() for s in self.scatterSeries[:len(self.actionlist)]]
        tables = [node.GetTable() for node in tableNodes]

        modifyStates = [node.StartModify() for node in tableNodes]

        try:
            for i, table in enumerate(tables):
                x = float(tsne_L[i, 0])
                y = float(tsne_L[i, 1])

                table.SetValue(0, 0, x)
                table.SetValue(1, 0, x)
                table.SetValue(0, 1, y)
                table.SetValue(1, 1, y)

        finally:
            for node, state in zip(tableNodes, modifyStates):
                node.EndModify(state)
            if chartNode:
                chartNode.EndModify(mChart)
            scene.EndState(scene.BatchProcessState)

    def updateTsne(self):

        # Update tsne
        if self.ui.tsneCheckBox.isChecked():
            tsne_L = TSNE(random_state=123,n_components=2,verbose=0,perplexity=40,max_iter=300).fit_transform(self.Ldpp)

            scene = slicer.mrmlScene
            plotLogic = slicer.modules.plots.logic()
            chartNode = self.plotChartNode
            scene.StartState(scene.BatchProcessState)
            mChart = chartNode.StartModify() if chartNode else None

            tableNodes = [s.GetTableNode() for s in self.scatterSeries[:len(self.actionlist)]]
            tables = [node.GetTable() for node in tableNodes]

            modifyStates = [node.StartModify() for node in tableNodes]

            try:
                for i, table in enumerate(tables):
                    x = float(tsne_L[i, 0])
                    y = float(tsne_L[i, 1])

                    table.SetValue(0, 0, x)
                    table.SetValue(1, 0, x)
                    table.SetValue(0, 1, y)
                    table.SetValue(1, 1, y)

            finally:
                for node, state in zip(tableNodes, modifyStates):
                    node.EndModify(state)
                if chartNode:
                    chartNode.EndModify(mChart)
                scene.EndState(scene.BatchProcessState)

    def onScatterSelected(self, mrmlPlotSeriesIDs, selectionCol):
        # pick ONE point to keep (last selected among all series)

        if mrmlPlotSeriesIDs.GetNumberOfValues()>0:
            keepSeriesID = None
            for i in range(mrmlPlotSeriesIDs.GetNumberOfValues()):
                seriesID = mrmlPlotSeriesIDs.GetValue(i)
                if seriesID!=self.currentSeriesID:
                    ids = vtk.vtkIdTypeArray.SafeDownCast(selectionCol.GetItemAsObject(i))
                    if ids and ids.GetNumberOfValues() > 0:
                        keepSeriesID = seriesID
                        keepPointId = int(ids.GetValue(ids.GetNumberOfValues() - 1))
                        self.currentSeriesID = keepSeriesID
                        break

            if keepSeriesID is None:
                return

            plotViewNode = self.plotWidget.mrmlPlotViewNode()     # key fix :contentReference[oaicite:3]{index=3}
            chartNode = plotViewNode.GetPlotChartNode()           # :contentReference[oaicite:4]{index=4}
            if not chartNode:
                return

            self.plotIndex = chartNode.GetPlotSeriesNodeIndexFromID(keepSeriesID)
            if self.plotIndex < 0:
                return

            chart = self.plotView.chart()  # vtkChartXY
            if not chart:
                return

            # clear selection in ALL plots
            empty = vtk.vtkIdTypeArray()
            for pi in range(chart.GetNumberOfPlots()):
                p = chart.GetPlot(pi)
                if p:
                    p.SetSelection(empty)

            # set selection for the kept point
            sel = vtk.vtkIdTypeArray()
            sel.InsertNextValue(keepPointId)
            p = chart.GetPlot(self.plotIndex)
            if p:
                p.SelectableOn()
                p.SetSelection(sel)

            # refresh
            try:
                self.plotView.renderWindow().Render()
            except Exception:
                pass
            
            self.actionIdxSave = []
            self.actionIdxSave.append(self.actionIdx)
            self.actionIdx = self.embID[self.plotIndex]
            self.process()

    def onNextButton(self):
        """
        Next
        """

        if not self.actionlist[self.actionIdx].info['classified']:
            self.actionlist[self.actionIdx].info['classified']=True

        # Update actionIdx
        if not self.settings['classification_multislice']:
            if self.settings['clustering']:
                self.actionIdxSave = [self.actionIdx]
                for i in range(self.actionIdx+1, len(self.actionlist)):
                    if self.actionlist[i].info['label']['batch']:
                        self.actionIdx = i
                        print('actionIdx123', self.actionIdx, i)
                        break
            else:
                self.actionIdxSave = [self.actionIdx]
                if self.actionIdx < len(self.actionlist):
                    self.actionIdx = self.actionIdx + 1
        else:
            self.actionIdxSave = []
            #for i in range(self.actionIdx+1, len(self.actionlist)):
            for i in range(self.actionIdx+1, len(self.actionlist)):
                #if self.actionlist[i].imagename != self.actionlist[self.actionIdx].imagename:
                if (self.actionlist[i].imagename != self.actionlist[self.actionIdx].imagename) or (self.actionlist[i].refinename != self.actionlist[self.actionIdx].refinename):
                    self.actionIdx=i
                    self.actionIdxSave.append(i-1)
                    break
                else:
                    self.actionlist[i].info['classified']=True
                self.actionIdxSave.append(i-1)

        self.process()
        
    def onBackButton(self):
        """
        Back
        """

        if self.actionlist[self.actionIdx].info['classified']:
            self.actionlist[self.actionIdx].info['classified']=False
           
        if not self.settings['classification_multislice']:
            #self.actionIdxSave = [self.actionIdx]
            if self.actionIdx > 0:
                self.actionIdx = self.actionIdx - 1
                self.actionlist[self.actionIdx].status = 'open'
                self.actionIdxSave = []

        self.process()

    def onNumberButton(self, number):
        if number>0 and number<=len(self.colorLabels):
            self.segmentEditorNode.SetSelectedSegmentID(self.colorLabels[number-1])
                
    def onMIP(self):
        print('onMIP', self.mip)
        if self.mip:
            sliceNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed")
            appLogic = slicer.app.applicationLogic()
            sliceLogic = appLogic.GetSliceLogic(sliceNode)
            sliceLayerLogic = sliceLogic.GetBackgroundLayer()
            reslice = sliceLayerLogic.GetReslice()
            reslice.SetSlabModeToMax()
            reslice.SetSlabNumberOfSlices(20) # mean of 10 slices will computed
            reslice.SetSlabSliceSpacingFraction(0.3) # spacing between each slice is 0.3 pixel (total 10 * 0.3 = 3 pixel neighborhood)
            sliceNode.Modified()
            self.mip=False
        else:
            sliceNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed")
            appLogic = slicer.app.applicationLogic()
            sliceLogic = appLogic.GetSliceLogic(sliceNode)
            sliceLayerLogic = sliceLogic.GetBackgroundLayer()
            reslice = sliceLayerLogic.GetReslice()
            reslice.SetSlabModeToSum()
            reslice.SetSlabNumberOfSlices(1) # mean of 10 slices will computed
            reslice.SetSlabSliceSpacingFraction(0.3) # spacing between each slice is 0.3 pixel (total 10 * 0.3 = 3 pixel neighborhood)
            sliceNode.Modified()
            self.mip=True
            
            
    def onToggleVisibility(self):
        segmentationNode=slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        segmentationDisplayNode=segmentationNode.GetDisplayNode()
        vis = segmentationDisplayNode.GetVisibility()
        segmentationDisplayNode.SetVisibility(not vis)

    def onToggleUncertainty(self):
        if self.foregroundOpacity>0.0:
            slicer.util.setSliceViewerLayers(foregroundOpacity=0.0)
            self.foregroundOpacity = 0.0
        else:
            slicer.util.setSliceViewerLayers(foregroundOpacity=0.5)
            self.foregroundOpacity = 0.5



    def onToggleROIVisibility(self):
        roiNodes=slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
        for roiNode in roiNodes:
            roiDisplayNode=roiNode.GetDisplayNode()
            vis = roiDisplayNode.GetVisibility()
            roiDisplayNode.SetVisibility(not vis)

    def onToggleThreshold(self):
        thr = self.segmentEditorNode.GetMasterVolumeIntensityMask()
        self.segmentEditorNode.SetMasterVolumeIntensityMask(not thr)
            
    def onSelectLabel(self):
        ras=[0,0,0]
        self.crosshairNode.GetCursorPositionRAS(ras)
        sliceViewLabel = "Red"
        sliceViewWidget = slicer.app.layoutManager().sliceWidget(sliceViewLabel)
        segmentationsDisplayableManager = sliceViewWidget.sliceView().displayableManagerByClassName("vtkMRMLSegmentationsDisplayableManager2D")
        segmentationNode=slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        segmentIds = vtk.vtkStringArray()
        segmentationsDisplayableManager.GetVisibleSegmentsForPosition(ras, segmentationNode.GetDisplayNode(), segmentIds)
        for idIndex in range(segmentIds.GetNumberOfValues()):
            segment = segmentationNode.GetSegmentation().GetSegment(segmentIds.GetValue(idIndex))
            self.segmentEditorNode.SetSelectedSegmentID(segment.GetName())

    def onSelectPositive(self):
        #print('onSelectPositive', self.plotIndex)
        a = self.actionlist[self.plotIndex]
        a.info['color'] = (0.0,0.0,1.0)
        a.info['label']['positive'] = not a.info['label']['positive']
        a.info['label']['negative'] = False
        #print('colorpos', a.info['color'])
        self.updateBatch()

    def onSelectNegative(self):
        #print('onSelectNegative', self.plotIndex)
        a = self.actionlist[self.plotIndex]
        a.info['color'] = (1.0,0.0,0.0)
        a.info['label']['negative'] = not a.info['label']['negative']
        a.info['label']['positive'] = False
        #print('colorneg', a.info['color'])
        self.updateBatch()

    def onTogglePseudo(self):
        self.selectPseudo = not self.selectPseudo
        self.ui.pseudoCheckBox.setChecked(self.selectPseudo)


    def setWindowLevel(self, window=800, level=250):
        self.volumeNode.GetDisplayNode().SetAutoWindowLevel(False)
        self.volumeNode.GetDisplayNode().SetWindowLevel(window, level)
            
    def onStopButton(self):
        """
        Stop
        """
        self.ui.startButton.setEnabled(True)
        self.ui.progressBar.setValue(0)
        slicer.mrmlScene.Clear()
        pass

    def onPseudoCheckBox(self, checked):
        """
        Pseudo
        """
        self.selectPseudo = checked
    
    
    
    def updateAlpha(self):
        self.alpha = self.ui.alphaBox.value
        self.updateBatch()
    
    def updateBeta(self):
        self.beta = self.ui.betaBox.value
        self.updateBatch()

    def updateGamma(self):
        self.gamma = self.ui.gammaBox.value
        self.updateBatch()

    def updateDelta(self):
        self.delta = self.ui.deltaBox.value
        self.updateBatch()

    def updateTheta(self):
        self.theta = self.ui.thetaBox.value
        idx = self.ui.strataBox.currentIndex
        #print('idx123', idx)
        self.strata_weight[idx] = self.theta
        #print('strata_weight123', self.strata_weight)
        self.updateBatch()
    
    def updateStrata(self):
        self.strata = self.ui.strataBox.currentText
        idx = self.ui.strataBox.currentIndex
        self.theta = self.strata_weight[idx]
        self.ui.thetaBox.setValue(self.theta)
        self.updateStrataMarker()


    def reslice_func(self, volumeNode, resolution=0.25, name='resliced'):
        
        global res
        
        def waitUntilReslice(timeout, res, period=0.25):
              mustend = time.time() + timeout
              while time.time() < mustend:
                if (res.GetStatusString() == 'Completed') or \
                    (res.GetStatusString() == 'Completing'): # I want to use only "Completed"
                    print(res.GetStatusString())
                    return True
                time.sleep(period)
              return False

        parameters = {
            "outputPixelSpacing":"{:},{:},{:}".format(*[resolution]*3),
            "InputVolume":volumeNode.GetID(),
            "interpolationMode":'bspline',
            "referenceVolume": volumeNode.GetID(),
            "OutputVolume":volumeNode.GetID()}
        
        res = slicer.cli.run(slicer.modules.resamplescalarvolume, None, parameters)
        
        waitUntilReslice(60, res)
        

        if(not volumeNode == None):
               volumeNode.GetDisplayNode().SetAutoWindowLevel(0)
               volumeNode.GetDisplayNode().SetWindowLevel(800,100)
    
        return volumeNode
    
    def updateBatch(self):

        def make_psd(K, eps=1e-8):
            K = 0.5 * (K + K.T)                   # symmetrize
            w, V = np.linalg.eigh(K)              # eigen-decomp (K should be symmetric)
            w = np.clip(w, eps, None)             # clip negatives to small positive
            return (V * w) @ V.T                  # V diag(w) V^T
        
        def build_L_from_K(K_psd, r, pi, uc, sweight, alpha=1.0, beta=1.0, gamma=1.0):
            z = (alpha*r - beta*pi + gamma*uc + sweight).astype(np.float64)
            z = np.clip(z, -40, 40)
            q = np.exp(z, dtype=np.float64)
            return (q[:, None] * K_psd) * q[None, :]

        idx_pos=[]
        idx_neg=[]
        for a in self.actionlist:
            idx_pos.append(a.info['label']['positive'])
            idx_neg.append(a.info['label']['negative'])
        idx_pos = np.array(idx_pos)
        idx_neg = np.array(idx_neg)

        #pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ submodlib

        fp_manual = os.path.dirname(self.settings['fip_actionlist'])
        fip_grad = os.path.join(fp_manual, "GRADMat.npy")
        if not os.path.isfile(fip_grad):
            Mgrad = np.random.rand(len(self.actionlist), len(self.actionlist))
            M_psd = make_psd(Mgrad)  
        else:
            Mgrad = np.load(fip_grad)
            MgradN = (Mgrad+1)/2
            M_psd = make_psd(MgradN)  

        try:
            from submodlib.functions.logDeterminantConditionalMutualInformation import LogDeterminantConditionalMutualInformationFunction
        except:
            pass
            # pip install scikit-learn
            #slicer.util.pip_install("-i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ submodlib")
            #from submodlib.functions.logDeterminantConditionalMutualInformation import LogDeterminantConditionalMutualInformationFunction
            slicer.util.pip_install("/Users/bfollmer/Downloads/submodlib-1.1.5-cp39-cp39-macosx_11_0_universal2.whl")
            from submodlib.functions.logDeterminantConditionalMutualInformation import LogDeterminantConditionalMutualInformationFunction

        try:
            from dppy.finite_dpps import FiniteDPP
        except:
            slicer.util.pip_install("dppy")
            from dppy.finite_dpps import FiniteDPP

        from scipy.stats import zscore
        uc = []
        if 'uc' in self.actionlist[0].info:
            uc=[]
            for i, action in enumerate(self.actionlist):
                uc.append(action.info['uc'])
            uc = np.array(uc)
            uc = scale_between_quantiles(uc)
        else:
            uc=[]
            for i, action in enumerate(self.actionlist):
                uc.append(0.5)
            uc = np.array(uc)

        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        delta = self.delta
        
        delta_scale = np.trace(M_psd) / M_psd.shape[0]
        K_psd = (1-delta) * M_psd + delta_scale * delta * np.eye(M_psd.shape[0])

        if idx_pos.sum()>0:
            K_pos = K_psd[:,idx_pos]
            r = K_pos.max(axis=1)
        else: 
            r = np.zeros(K_psd.shape[0])

        if idx_neg.sum()>0:
            K_neg = K_psd[:,idx_neg]
            pi = K_neg.max(axis=1)
        else: 
            pi = np.zeros(K_psd.shape[0])

        theta = self.theta
        batch_size = self.settings['batch_size']
        thr_fg = 0.00001

        sweight = []
        for a in self.actionlist:
            wa = 0
            for i,st in enumerate(self.strata_info):
                if st in a.info['strata_info']:
                    wa = wa + self.strata_weight[i]
            sweight.append(wa)


        Ldpp = build_L_from_K(K_psd, r, pi, uc, sweight, alpha=alpha, beta=beta, gamma=gamma)
        self.Ldpp = Ldpp

        dpp = FiniteDPP(kernel_type="likelihood", L=Ldpp)
        rng = np.random.RandomState(0) 
        dpp.sample_exact_k_dpp(size=batch_size, random_state=rng)          # exact k-DPP sample
        batch = dpp.list_of_samples[-1]         # list of selected indices
        greedyIndices = np.array(batch)
        actions_sel = [self.actionlist[i] for i in greedyIndices]

        uc = []
        if 'uc' in self.actionlist[0].info:
            uc=[]
            for i, action in enumerate(self.actionlist):
                uc.append(action.info['uc'])
            uc = np.array(uc)
            uc = scale_between_quantiles(uc)            
        else:
            uc=[]
            for i, action in enumerate(self.actionlist):
                uc.append(0.5)
            uc = np.array(uc)

        print('uc123X', uc)
        for i, a in enumerate(self.actionlist):
            if not a.info['label']['positive'] and not a.info['label']['negative']:
                a.info['color'] = ( 1.0-float(uc[i]), 1.0-float(uc[i]), 1.0-float(uc[i]) )
        # Set color of selected samples
        for a in actions_sel:
            if not a.info['label']['positive'] and not a.info['label']['negative']:
                a.info['color'] = (0.0,1.0, 0.0)
        
        # Set batch label
        for a in self.actionlist:
            a.info['label']['batch']=False
        for a in actions_sel:
            a.info['label']['batch']=True

        # Set strata color
        for i,a in enumerate(self.actionlist):
            setScatterColor(self.scatterSeries[i], a.info['color'])

        if 'uc' in actions_sel[0].info:
            uc_mean_sel = np.mean([a.info['uc'] for a in actions_sel])
            uc_mean_all = np.mean([a.info['uc'] for a in self.actionlist])
            uc_ratio = uc_mean_sel/uc_mean_all
            txtuc = f"Uncertainty average: {uc_ratio:.2f}\n"
        else:
            txtuc = ''

        # Update tsne
        self.updateTsne()

        # Compute fg
        if 'fg' in actions_sel[0].info:
            fgs = np.zeros(len(actions_sel[0].info['fg']))
            for i, a in enumerate(actions_sel):
                fg = np.array(a.info['fg'])
                fg = (fg>thr_fg)*1.0
                fgs = fgs + fg
            fg_ratio = fgs / len(actions_sel)   
            txtfg = ""
            for i,lab in enumerate(a.label[0:-2]):
                txtfg = txtfg + f"{lab[1]}:{fg_ratio[i]}\n"
        else:
            txtfg = ""

        # Show batch information
        txt = ""
        txt = txt + txtuc
        txt = txt + txtfg
        
        self.ui.textBatch.setPlainText(txt)
        # Save actionlist
        ALAction.save(self.settings['fip_actionlist'], self.actionlist)

    def updateStrataMarker(self):

        scene = slicer.mrmlScene
        plotLogic = slicer.modules.plots.logic()

        #chartNode = plotLogic.GetPlotChartNode()  # if you use a specific chart, use that node instead
        chartNode = self.plotChartNode
        scene.StartState(scene.BatchProcessState)
        mChart = chartNode.StartModify() if chartNode else None
        try:
            for i, a in enumerate(self.actionlist):
                if 'strata_info' in a.info:
                    if self.strata in a.info['strata_info']:
                        markerSize = 12
                        ps = self.scatterSeries[i]
                        m = ps.StartModify()
                        ps.SetMarkerSize(markerSize)
                        ps.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleDiamond)
                        ps.EndModify(m)
                    else:
                        markerSize = 12
                        ps = self.scatterSeries[i]
                        m = ps.StartModify()
                        ps.SetMarkerSize(markerSize)
                        ps.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleCircle)
                        ps.EndModify(m)
        finally:
            if chartNode:
                chartNode.EndModify(mChart)
            scene.EndState(scene.BatchProcessState)


    def updateSegmentation(self):
        # Save segmentation if exist
        if len(self.actionIdxSave) > 0:
            #print("actionIdxSave345", self.actionIdxSave[0])
            action = self.actionlist[self.actionIdxSave[0]]
            if self.pseudoNode is not None:
                if action.fip_refine is None:
                    fip_refine = os.path.join(self.settings['fp_refine'], action.refinename)
                else:
                    fip_refine = action.fip_refine

                #print('actionIdxSave123', self.actionIdxSave)
                #print('CreateLM', 'LabelMap_' + action.imagename)
                labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'LabelMap_' + action.imagename)
                segmentIds = self.pseudoNode.GetSegmentation().GetSegmentIDs()
                slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsToLabelmapNode(self.pseudoNode, segmentIds, labelmapVolumeNode, self.volumeNode)
                # Check if pseudo label was used
                if action.fip_pseudo is None:
                    fip_pseudo = os.path.join(self.settings['fp_pseudo'], action.pseudoname)
                else:
                    fip_pseudo = action.fip_pseudo
                pseudo_used = os.path.isfile(fip_pseudo)
                label_ignore = self.colorLabels.index('ignore')
                label_pseudo = self.colorLabels.index('pseudo')
                #print('label_ignore123', label_ignore)
                if self.settings['default_ignore'] and not pseudo_used:
                    print('func01')
                    labelmapArray = slicer.util.arrayFromVolume(labelmapVolumeNode)
                    labelmapArray = labelmapArray-1
                    labelmapArray[labelmapArray==-1]=label_ignore
                    labelmapVolumeNodeMod = slicer.util.addVolumeFromArray(labelmapArray)
                    labelmapVolumeNodeMod.SetSpacing(labelmapVolumeNode.GetSpacing())
                    labelmapVolumeNodeMod.SetOrigin(labelmapVolumeNode.GetOrigin())
                    ijkdirs = [[0,0,0],[0,0,0],[0,0,0]]
                    labelmapVolumeNode.GetIJKToRASDirections(ijkdirs)
                    labelmapVolumeNodeMod.SetIJKToRASDirections(ijkdirs)
                    slicer.util.saveNode(labelmapVolumeNodeMod, fip_refine)
                    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
                    slicer.mrmlScene.RemoveNode(labelmapVolumeNodeMod)

                elif self.settings['default_ignore'] and pseudo_used:
                    pseudo_im = sitk.ReadImage(fip_pseudo)
                    pseudo_arr = sitk.GetArrayFromImage(pseudo_im)
                    labelmap_arr = slicer.util.arrayFromVolume(labelmapVolumeNode).astype(np.int16)
                    labelmap_arr[labelmap_arr==0]=1
                    labelmap_arr = labelmap_arr-1
                    diff = labelmap_arr != pseudo_arr
                    idx_class_changed = np.where(diff==True)
                    #idx_pseudo = np.where(labelmap_arr==label_pseudo)
                    if self.selectPseudo:
                        diff_pseudo = np.ones(diff.shape)*True
                        diff_pseudo[action.bboxLbsOrg[0]:action.bboxUbsOrg[0], action.bboxLbsOrg[1]:action.bboxUbsOrg[1], action.bboxLbsOrg[2]:action.bboxUbsOrg[2]] = diff[action.bboxLbsOrg[0]:action.bboxUbsOrg[0], action.bboxLbsOrg[1]:action.bboxUbsOrg[1], action.bboxLbsOrg[2]:action.bboxUbsOrg[2]]
                        idx_pseudo = np.where(diff_pseudo==False)
                    else:
                        idx_pseudo = np.where(labelmap_arr==label_pseudo)

                    labelmapOut = np.ones(labelmap_arr.shape)*label_ignore
                    labelmapOut[idx_class_changed] = labelmap_arr[idx_class_changed]
                    labelmapOut[idx_pseudo] = pseudo_arr[idx_pseudo]
                    labelmapVolumeNodeMod = slicer.util.addVolumeFromArray(labelmapOut)
                    labelmapVolumeNodeMod.SetSpacing(labelmapVolumeNode.GetSpacing())
                    labelmapVolumeNodeMod.SetOrigin(labelmapVolumeNode.GetOrigin())
                    ijkdirs = [[0,0,0],[0,0,0],[0,0,0]]
                    labelmapVolumeNode.GetIJKToRASDirections(ijkdirs)
                    labelmapVolumeNodeMod.SetIJKToRASDirections(ijkdirs)
                    slicer.util.saveNode(labelmapVolumeNodeMod, fip_refine)
                    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
                    #print('remove12345', labelmapVolumeNode.GetName())
                    slicer.mrmlScene.RemoveNode(labelmapVolumeNodeMod)
                    #print("RemoveNode123")
                else:
                    print('func03')
                    labelmapArray = slicer.util.arrayFromVolume(labelmapVolumeNode)
                    labelmapArray = labelmapArray-1
                    labelmapArray[labelmapArray==-1]=0
                    labelmapVolumeNodeMod = slicer.util.addVolumeFromArray(labelmapArray)
                    labelmapVolumeNodeMod.SetSpacing(labelmapVolumeNode.GetSpacing())
                    labelmapVolumeNodeMod.SetOrigin(labelmapVolumeNode.GetOrigin())
                    labelmapVolumeNodeMod.SetDirection(labelmapVolumeNode.GetDirection())
                    slicer.util.saveNode(labelmapVolumeNodeMod, fip_refine)
                    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
                    #print('remove1234', labelmapVolumeNode.GetName())
                    slicer.mrmlScene.RemoveNode(labelmapVolumeNodeMod)


    def process(self):
        import time
        start = time.time()
        self.updateSegmentation()

        # Save actionlist
        if self.actionIdx > 0:
            #print('self.actionIdxSave123', self.actionIdxSave, self.selection)
            for idxSave in self.actionIdxSave:
                #self.actionlist[idxSave].status = 'solved'
                if not self.selection:
                    self.actionlist[idxSave].status = 'solved'
                if self.settings['classification']:
                    self.actionlist[idxSave].info['annotated']=False
                else:
                    self.actionlist[idxSave].info['annotated']=True
            ALAction.save(self.settings['fip_actionlist'], self.actionlist)

        if not self.settings['clustering']:
            idx_start = self.actionIdx
            for i in range(idx_start, len(self.actionlist)+1):
                self.actionIdx = i
                if self.actionIdx==len(self.actionlist):
                    self.onStopButton()
                    return
                action = self.actionlist[self.actionIdx]
                #print('action.status12', action.status)
                if action.status == 'open':
                    break
        else:
            action = self.actionlist[self.actionIdx]

        # Update progressbar
        if not self.selection:
            self.ui.progressBar.setValue((self.actionIdx/len(self.actionlist))*100)
        
        # Set color
        colorNode = slicer.util.loadColorTable(self.settings['fip_colors'])
        for i in range(colorNode.GetNumberOfColors()):
            cname = colorNode.GetColorName(i)
            colorNode.SetColorName(i, cname.replace(' ', '_'))
        
        # Update pseudoNode
        if slicer.mrmlScene.GetFirstNodeByName(action.pseudoname) is None:
            if self.pseudoNode is None:
                if action.fip_pseudo is None:
                    fip_pseudo = os.path.join(self.settings['fp_pseudo'], action.pseudoname)
                else:
                    fip_pseudo = action.fip_pseudo
                if os.path.isfile(fip_pseudo):
                    labelmapVolumeNode = slicer.util.loadLabelVolume(fip_pseudo)
                    labelmapArray = slicer.util.arrayFromVolume(labelmapVolumeNode)
                    labelmapArray = labelmapArray+1
                    slicer.util.updateVolumeFromArray(labelmapVolumeNode, labelmapArray)
                    self.pseudoNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", action.pseudoname)
                    segmentIds = vtk.vtkStringArray()
                    for label in action.label:
                        segmentId = self.pseudoNode.GetSegmentation().AddEmptySegment(str(label[1]))
                        segment = self.pseudoNode.GetSegmentation().GetSegment(segmentId)
                        segment.SetColor(label[2][0], label[2][1], label[2][2])
                        segment.SetName(label[1])                    
                        self.pseudoNode.GetSegmentation().SetSegmentIndex(segmentId, label[0])
                        segmentIds.InsertNextValue(segmentId)
                    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, self.pseudoNode, segmentIds)
                    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
                    
                    # Disable backround visibility
                    info_dict = action.info
                    #print("backround_visibility123")
                    segmentationNode=slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
                    segmentationDisplayNode=segmentationNode.GetDisplayNode()
                    #print("backround_visibility12378", segmentationDisplayNode)

                    if 'backround_visibility' in action.info:
                        #print("backround_visibility1234")
                        segmentationNode=slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
                        segmentationDisplayNode=segmentationNode.GetDisplayNode()
                        segmentationDisplayNode.SetSegmentVisibility('background', action.info['backround_visibility'])

                else:
                    self.pseudoNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", action.pseudoname)
                    for label in action.label:
                        segmentId = self.pseudoNode.GetSegmentation().AddEmptySegment(str(label[1]))
                        segment = self.pseudoNode.GetSegmentation().GetSegment(segmentId)
                        segment.SetColor(label[2][0], label[2][1], label[2][2])
                        segment.SetName(label[1])                    
                        self.pseudoNode.GetSegmentation().SetSegmentIndex(segmentId, label[0])

            else:
                if action.fip_pseudo is None:
                    fip_pseudo = os.path.join(self.settings['fp_pseudo'], action.pseudoname)
                else:
                    fip_pseudo = action.fip_pseudo
                slicer.mrmlScene.RemoveNode(self.pseudoNode)
                #print("segmentationDisplayNode1234")
                if os.path.isfile(fip_pseudo):
                    #self.pseudoNode = slicer.util.loadSegmentation(fip_pseudo, properties={'fileType': 'SegmentationFile', 'name': action.pseudoname, 'colorNodeID': colorNode.GetID(), 'show': True})
                    labelmapVolumeNode = slicer.util.loadLabelVolume(fip_pseudo)
                    labelmapArray = slicer.util.arrayFromVolume(labelmapVolumeNode)
                    labelmapArray = labelmapArray+1
                    slicer.util.updateVolumeFromArray(labelmapVolumeNode, labelmapArray)
                    self.pseudoNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", action.pseudoname)
                    segmentIds = vtk.vtkStringArray()
                    for label in action.label:
                        segmentId = self.pseudoNode.GetSegmentation().AddEmptySegment(str(label[1]))
                        segment = self.pseudoNode.GetSegmentation().GetSegment(segmentId)
                        segment.SetColor(label[2][0], label[2][1], label[2][2])
                        segment.SetName(label[1])
                        self.pseudoNode.GetSegmentation().SetSegmentIndex(segmentId, label[0])
                        segmentIds.InsertNextValue(segmentId)
                    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, self.pseudoNode, segmentIds)
                    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
                    
                    # Disable backround visibility
                    info_dict = action.info
                    if 'backround_visibility' in action.info:
                        segmentationNode=slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
                        segmentationDisplayNode=segmentationNode.GetDisplayNode()
                        print("segmentationDisplayNode123", segmentationDisplayNode)
                        segmentationDisplayNode.SetSegmentVisibility('background', action.info['backround_visibility'])

                else:
                    slicer.mrmlScene.RemoveNode(self.pseudoNode)
                    #self.pseudoNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", action.pseudoname)
                    self.pseudoNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", action.pseudoname)
                    for label in action.label:
                        segmentId = self.pseudoNode.GetSegmentation().AddEmptySegment(str(label[1]))
                        segment = self.pseudoNode.GetSegmentation().GetSegment(segmentId)
                        segment.SetColor(label[2][0], label[2][1], label[2][2])
                        segment.SetName(label[1])                    
                        self.pseudoNode.GetSegmentation().SetSegmentIndex(segmentId, label[0])
            self.ui.segmentEditorWidget.setSegmentationNode(self.pseudoNode)

        # Update volumeNode   
        if slicer.mrmlScene.GetFirstNodeByName(action.imagename) is None:
            if self.volumeNode is None:
                #print('action.fip_image123', action.fip_image)
                if action.fip_image is None:
                    # Check if image is in the fp_images folder or in a subfolder
                    fip_image = os.path.join(self.settings['fp_images'], action.imagename)
                    if not os.path.isfile(fip_image):
                        fp_images_sub = glob(self.settings['fp_images'] + '/*')
                        for fp_sub in fp_images_sub:
                            fip_image = os.path.join(fp_sub, action.imagename)
                            if os.path.isfile(fip_image):
                                break
                else:
                    fip_image = action.fip_image
                #print('fip_image1234', fip_image)
                self.volumeNode = slicer.util.loadVolume(fip_image, properties={'name': action.imagename, 'show': True})
            else:
                #print('fip_image12345', action.fip_image)
                if action.fip_image is None:
                    # Check if image is in the fp_images folder or in a subfolder
                    fip_image = os.path.join(self.settings['fp_images'], action.imagename)
                    #print('fip_image123', fip_image)
                    if not os.path.isfile(fip_image):
                        fp_images_sub = glob(self.settings['fp_images'] + '/*')
                        #print('fp_images_sub123', fp_images_sub)
                        for fp_sub in fp_images_sub:
                            fip_image = os.path.join(fp_sub, action.imagename)
                            if os.path.isfile(fip_image):
                                break
                else:
                    fip_image = action.fip_image
                    
                slicer.mrmlScene.RemoveNode(self.volumeNode)
                self.volumeNode = slicer.util.loadVolume(fip_image, properties={'name': action.imagename, 'show': True})
            if 'WindowLevel' in self.settings.settingsDict:
                self.setWindowLevel(window=self.settings['WindowLevel']['window'], level=self.settings['WindowLevel']['level'])
            else:
                self.setWindowLevel()
            self.ui.segmentEditorWidget.setSourceVolumeNode(self.volumeNode)
            # Reset ROI
            for node in self.roiNode:
                slicer.mrmlScene.RemoveNode(node)
            self.roiNode=[]
        else:
            self.ui.segmentEditorWidget.setSourceVolumeNode(self.volumeNode)

        # Update uncertainty map
        if action.ucname is not None:
            ucname = action.ucname.split('.')[0] + '_uc' + '.nii.gz'
            #print('ucname678', ucname)
            if slicer.mrmlScene.GetFirstNodeByName(ucname) is None:
                if self.ucNode is None:
                    #print('action.fip_image123', action.fip_image)
                    if action.fip_uc is None:
                        # Check if image is in the fp_images folder or in a subfolder
                        fip_uc = os.path.join(self.settings['fp_uc'], action.ucname)
                        #print('fip_uc123456', fip_uc)
                        if not os.path.isfile(fip_uc):
                            fip_uc_sub = glob(self.settings['fp_uc'] + '/*')
                            for fp_sub in fip_uc_sub:
                                fip_uc = os.path.join(fp_sub, action.ucname)
                                if os.path.isfile(fip_uc):
                                    break
                    else:
                        fip_uc = action.fip_uc
                    if os.path.isfile(fip_uc):
                        self.ucNode = slicer.util.loadVolume(fip_uc, properties={'name': ucname, 'show': True})
                    else:
                        self.ucNode = None
                else:
                    #print('fip_image12345', action.fip_image)
                    if action.fip_uc is None:
                        # Check if image is in the fp_images folder or in a subfolder
                        fip_uc = os.path.join(self.settings['fp_uc'], action.ucname)
                        if not os.path.isfile(fip_uc):
                            fp_uc_sub = glob(self.settings['fp_uc'] + '/*')
                            for fp_sub in fp_uc_sub:
                                fip_uc = os.path.join(fp_sub, action.ucname)
                                if os.path.isfile(fip_uc):
                                    break
                    else:
                        fip_uc = action.fip_uc
                        
                    slicer.mrmlScene.RemoveNode(self.ucNode)
                    self.ucNode = slicer.util.loadVolume(fip_uc, properties={'name': ucname, 'show': True})
                
            else:
                pass
            # Update window level
            if self.ucNode is not None:
                self.ucNode.GetDisplayNode().SetAutoWindowLevel(False)
                self.ucNode.GetDisplayNode().SetWindowLevel(800, 100)
            

            slicer.util.setSliceViewerLayers(foreground=self.ucNode,background=self.volumeNode)
            self.foregroundOpacity = 0.5
            slicer.util.setSliceViewerLayers(foregroundOpacity=self.foregroundOpacity)
            if self.ucNode is not None:
                self.ucNode.GetDisplayNode().SetAndObserveColorNodeID('vtkMRMLColorTableNodeIron')
            #self.ucNode.GetDisplayNode().SetThreshold(10,900)
            self.pseudoNode.GetDisplayNode().SetVisibility2DFill(False)
            self.pseudoNode.GetDisplayNode().SetVisibility2DOutline(True)

        # Update layoutManager
        if self.settings['classification_multislice']:
            slice_vis = 0
        else:
            if action.slice is not None:
                slice_vis = action.slice
            else:
                slice_vis = int((action.bboxLbsOrg[0]+action.bboxUbsOrg[0])/2)

        #offset = self.redLogic.GetSliceOffset()
        origen = self.volumeNode.GetOrigin()
        spacing = self.volumeNode.GetSpacing()
        img = self.volumeNode.GetImageData()
        dimension = img.GetDimensions()
        
        direction = np.eye(3)
        self.volumeNode.GetIJKToRASDirections(direction)
        if direction[0,0]==-1 and direction[1,1]==-1 and direction[2,2]==1:
            #slice_vis_offset = slice_vis

            print('fp_images', self.settings['fp_images'])

            # Correct in KITS dataset
            if 'KITS' in self.settings['fp_images']:
                slice_vis_offset = dimension[2]-slice_vis-1
                #slice_vis_offset = slice_vis
                axis = 1
            elif 'ASOCA' in self.settings['fp_images']:
                slice_vis_offset = slice_vis
                axis = 1
            else:
                slice_vis_offset = slice_vis
                axis = 1
            #slice_vis_offset = slice_vis
            #axis = 1
        elif direction[0,0]==-1 and direction[1,1]==1 and direction[2,2]==-1:
            #slice_vis_offset = dimension[2]-slice_vis-1
            slice_vis = int((action.bboxLbsOrg[0]+action.bboxUbsOrg[0])/2)
            slice_vis_offset = dimension[2]-slice_vis-1
            axis = -1
        else:

            # !!! Check if this is correct!
            slice_vis_offset = dimension[2]-slice_vis-1
            axis = -1

        offset = origen[2] + axis * slice_vis_offset * spacing[2]
        self.redLogic.SetSliceOffset(offset)
        
        # Set ROI
        if self.settings['show_roi']:
            #print('self.roiNode1235678', self.roiNode)
            if not self.settings['classification_multislice']:
                if len(self.roiNode)==0:
                    self.roiNode.append(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode"))
                # Set ROI center
                if action.dim==2:
                    dx = int((action.bboxUbsOrg[2]+action.bboxLbsOrg[2])/2)
                    dy = int((action.bboxUbsOrg[1]+action.bboxLbsOrg[1])/2)
                    dz = int((action.bboxUbsOrg[0]+action.bboxLbsOrg[0])/2)
                    ijkToRas = vtk.vtkMatrix4x4()
                    self.volumeNode.GetIJKToRASMatrix(ijkToRas)
                    if direction[2,2]>0:
                        position_Ijk=[dx, dy, dz, 1]
                        position_Ras=ijkToRas.MultiplyPoint(position_Ijk)
                        #print('position_Ras1235', position_Ras)
                        self.roiNode[0].SetCenter(position_Ras[0:3])
                    else:
                        #print('dz123', dz)
                        position_Ijk=[dx, dy, dimension[2]-dz-1, 1]
                        position_Ras=ijkToRas.MultiplyPoint(position_Ijk)
                        #print('position_Ras1234', position_Ras)
                        self.roiNode[0].SetCenter(position_Ras[0:3])
                    if direction[2,2]>0:
                        # !!! Check
                        position_Ijk=[action.bboxLbsOrg[1], action.bboxLbsOrg[2], dimension[2]-action.bboxLbsOrg[0]-1, 1]
                        bboxL=ijkToRas.MultiplyPoint(position_Ijk)
                        position_Ijk=[action.bboxUbsOrg[1], action.bboxUbsOrg[2], dimension[2]-action.bboxUbsOrg[0]-1, 1]
                        bboxU=ijkToRas.MultiplyPoint(position_Ijk)
                    else:
                        position_Ijk=[action.bboxLbsOrg[1], action.bboxLbsOrg[2], dimension[2]-action.bboxLbsOrg[0]-1, 1]
                        bboxL=ijkToRas.MultiplyPoint(position_Ijk)
                        position_Ijk=[action.bboxUbsOrg[1], action.bboxUbsOrg[2], dimension[2]-action.bboxUbsOrg[0]-1, 1]
                        bboxU=ijkToRas.MultiplyPoint(position_Ijk)

                    self.roiNode[0].SetSizeWorld([bboxU[0]-bboxL[0], bboxU[1]-bboxL[1], bboxU[2]-bboxL[2]])
                else:
                    #print('direction045', direction)
                    dx = int((action.bboxUbsOrg[2]+action.bboxLbsOrg[2])/2)
                    dy = int((action.bboxUbsOrg[1]+action.bboxLbsOrg[1])/2)
                    dz = int((action.bboxUbsOrg[0]+action.bboxLbsOrg[0])/2)
                    ijkToRas = vtk.vtkMatrix4x4()
                    self.volumeNode.GetIJKToRASMatrix(ijkToRas)
                    if direction[0,0]==-1 and direction[1,1]==-1 and direction[2,2]==1:
                        if 'ASOCA' in self.settings['fp_images']:
                            position_Ijk=[dx, dy, dz, 1]
                            print('here03')
                        else:
                            position_Ijk=[dx, dy, dimension[2]-dz-1, 1]
                        position_Ras=ijkToRas.MultiplyPoint(position_Ijk)
                        #print('position_Ras1234', position_Ras)
                        self.roiNode[0].SetCenter(position_Ras[0:3])
                    elif direction[0,0]==-1 and direction[1,1]==1 and direction[2,2]==-1:
                        position_Ijk=[dx, dy, dimension[2]-dz-1, 1]
                        position_Ras=ijkToRas.MultiplyPoint(position_Ijk)
                        #print('position_Ras1234', position_Ras)
                        self.roiNode[0].SetCenter(position_Ras[0:3])
                    else:
                        position_Ijk=[dx, dy, dz, 1]
                        position_Ras=ijkToRas.MultiplyPoint(position_Ijk)
                        #print('position_Ras1235', position_Ras)
                        self.roiNode[0].SetCenter(position_Ras[0:3])

                    # Set ROI size
                    # position_Ijk=[action.bboxLbsOrg[1], action.bboxLbsOrg[2], action.bboxLbsOrg[0], 1]
                    # bboxL=ijkToRas.MultiplyPoint(position_Ijk)
                    # position_Ijk=[action.bboxUbsOrg[1], action.bboxUbsOrg[2], action.bboxUbsOrg[0], 1]
                    # bboxU=ijkToRas.MultiplyPoint(position_Ijk)

                    if direction[2,2]>0:
                        #position_Ijk=[action.bboxLbsOrg[1], action.bboxLbsOrg[2], action.bboxLbsOrg[0], 1]
                        #bboxL=ijkToRas.MultiplyPoint(position_Ijk)
                        #position_Ijk=[action.bboxUbsOrg[1], action.bboxUbsOrg[2], action.bboxUbsOrg[0], 1]
                        #bboxU=ijkToRas.MultiplyPoint(position_Ijk)
                        # !!! Check
                        position_Ijk=[action.bboxLbsOrg[1], action.bboxLbsOrg[2], dimension[2]-action.bboxLbsOrg[0]-1, 1]
                        bboxL=ijkToRas.MultiplyPoint(position_Ijk)
                        position_Ijk=[action.bboxUbsOrg[1], action.bboxUbsOrg[2], dimension[2]-action.bboxUbsOrg[0]-1, 1]
                        bboxU=ijkToRas.MultiplyPoint(position_Ijk)
                        if 'ASOCA' in self.settings['fp_images']:
                            print('here02')
                            position_Ijk=[action.bboxLbsOrg[1], action.bboxLbsOrg[2], action.bboxLbsOrg[0]-1, 1]
                            bboxL=ijkToRas.MultiplyPoint(position_Ijk)
                            position_Ijk=[action.bboxUbsOrg[1], action.bboxUbsOrg[2], action.bboxUbsOrg[0]-1, 1]
                            bboxU=ijkToRas.MultiplyPoint(position_Ijk)
                    else:
                        position_Ijk=[action.bboxLbsOrg[1], action.bboxLbsOrg[2], dimension[2]-action.bboxLbsOrg[0]-1, 1]
                        bboxL=ijkToRas.MultiplyPoint(position_Ijk)
                        position_Ijk=[action.bboxUbsOrg[1], action.bboxUbsOrg[2], dimension[2]-action.bboxUbsOrg[0]-1, 1]
                        bboxU=ijkToRas.MultiplyPoint(position_Ijk)
                        if 'ASOCA' in self.settings['fp_images']:
                            print('here01')
                            position_Ijk=[action.bboxLbsOrg[1], action.bboxLbsOrg[2], action.bboxLbsOrg[0]-1, 1]
                            bboxL=ijkToRas.MultiplyPoint(position_Ijk)
                            position_Ijk=[action.bboxUbsOrg[1], action.bboxUbsOrg[2], action.bboxUbsOrg[0]-1, 1]
                            bboxU=ijkToRas.MultiplyPoint(position_Ijk)

                    self.roiNode[0].SetSizeWorld([bboxU[0]-bboxL[0], bboxU[1]-bboxL[1], bboxU[2]-bboxL[2]])
            else:
                #print('self.roiNode123', self.roiNode)
                for ac in self.actionlist:
                    if ac.imagename==action.imagename:
                        #if self.roiNode is None:
                        #    self.roiNode=[]
                        self.roiNode.append(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode"))
                        
                        # Set ROI center
                        if ac.dim==2:
                            dx = int((ac.bboxUbsOrg[2]+ac.bboxLbsOrg[2])/2)
                            dy = int((ac.bboxUbsOrg[1]+ac.bboxLbsOrg[1])/2)
                            dz = int((ac.bboxUbsOrg[0]+ac.bboxLbsOrg[0])/2)
                            ijkToRas = vtk.vtkMatrix4x4()
                            self.volumeNode.GetIJKToRASMatrix(ijkToRas)
                            position_Ijk=[dx, dy, dz, 1]
                            position_Ras=ijkToRas.MultiplyPoint(position_Ijk)
                            self.roiNode[-1].SetCenter(position_Ras[0:3])
                            # Set ROI size
                            position_Ijk=[ac.bboxLbsOrg[1], ac.bboxLbsOrg[2], ac.bboxLbsOrg[0], 1]
                            bboxL=ijkToRas.MultiplyPoint(position_Ijk)
                            position_Ijk=[ac.bboxUbsOrg[1], ac.bboxUbsOrg[2], ac.bboxUbsOrg[0], 1]
                            bboxU=ijkToRas.MultiplyPoint(position_Ijk)
                            self.roiNode[-1].SetSizeWorld([bboxU[0]-bboxL[0], bboxU[1]-bboxL[1], bboxU[2]-bboxL[2]])
                        else:
                            dx = int((ac.bboxUbsOrg[1]+ac.bboxLbsOrg[1])/2)
                            dy = int((ac.bboxUbsOrg[2]+ac.bboxLbsOrg[2])/2)
                            dz = int((ac.bboxUbsOrg[0]+ac.bboxLbsOrg[0])/2)
                            ijkToRas = vtk.vtkMatrix4x4()
                            self.volumeNode.GetIJKToRASMatrix(ijkToRas)
                            position_Ijk=[dx, dy, dz, 1]
                            print('position_Ijk123', position_Ijk)
                            position_Ras=ijkToRas.MultiplyPoint(position_Ijk)
                            print('position_Ras123', position_Ras)
                            self.roiNode[-1].SetCenter(position_Ras[0:3])
                            position_Ijk=[ac.bboxLbsOrg[1], ac.bboxLbsOrg[2], ac.bboxLbsOrg[0], 1]
                            bboxL=ijkToRas.MultiplyPoint(position_Ijk)
                            position_Ijk=[ac.bboxUbsOrg[1], ac.bboxUbsOrg[2], ac.bboxUbsOrg[0], 1]
                            bboxU=ijkToRas.MultiplyPoint(position_Ijk)
                            print('bboxL123', bboxL)
                            print('bboxU123', bboxU)
                            self.roiNode[-1].SetSizeWorld([bboxU[0]-bboxL[0], bboxU[1]-bboxL[1], bboxU[2]-bboxL[2]])
                
        # Update segment editor
        self.ui.segmentEditorWidget.setActiveEffectByName("Paint")
        
        # Fill by thresholding
        if 'threshold' in action.info:
            threshold = action.info['threshold']
            #self.segmentEditorNode.MasterVolumeIntensityMaskOn()
            self.segmentEditorNode.SetMasterVolumeIntensityMask(True)
            self.segmentEditorNode.SetSourceVolumeIntensityMaskRange(threshold[0], threshold[1])
        else:
            self.segmentEditorNode.SetMasterVolumeIntensityMask(False)
        
        if self.ucNode is not None:
            self.ucNode.GetDisplayNode().SetAutoThreshold(0)
            self.ucNode.GetDisplayNode().ApplyThresholdOn()
            self.ucNode.GetDisplayNode().SetThreshold(10,1000)
        #print('Endtime', time.time() - start)
        
    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                                self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

            # Compute inverted output (if needed)
            if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
                self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
                                    self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)


#
# XALabelerLogic
#

class XALabelerLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")

    def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


#
# XALabelerTest
#

class XALabelerTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_XALabeler1()

    def test_XALabeler1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('XALabeler1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = XALabelerLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
