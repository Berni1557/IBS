# -*- coding: utf-8 -*-
import os
import json
from collections import OrderedDict

# Settings
class Settings:
    def __init__(self):
        self.settingsDict=None
        
    def __getitem__(self, key):
        return self.settingsDict[key]

    def __setitem__(self, key, value):
        self.settingsDict[key] = value

    def readSettings(self, filepath_settings):
        """ Read settings from setting file

        :param filepath_settings: Filepath to settings file
        :type filepath_settings: str
        """
        
        # def _decode_list(data):
        #     rv = []
        #     for item in data:
        #         if isinstance(item, unicode):
        #             item = item.encode('utf-8')
        #         elif isinstance(item, list):
        #             item = _decode_list(item)
        #         elif isinstance(item, dict):
        #             item = _decode_dict(item)
        #         rv.append(item)
        #     return rv
            
        # def _decode_dict(data):
        #     rv = {}
        #     for key, value in data.iteritems():
        #         if isinstance(key, unicode):
        #             key = key.encode('utf-8')
        #         if isinstance(value, unicode):
        #             value = value.encode('utf-8')
        #         elif isinstance(value, list):
        #             value = _decode_list(value)
        #         elif isinstance(value, dict):
        #             value = _decode_dict(value)
        #         rv[key] = value
        #     return rv
    
        if os.path.isfile(filepath_settings):
            print('Reading setting from ' + filepath_settings)
            with open(filepath_settings) as f:
                #settings = json.load(f, object_hook=_decode_dict, object_pairs_hook=OrderedDict)
                settings = json.load(f)
                self.checkSettings(settings)
                settings = OrderedDict(settings)
                # CreateCACSTree
                #settings['CACSTree'] = CACSTree()
                #settings['CACSTree'].createTree(settings)
                self.settingsDict = settings
        else:
            print('Settings file:' + filepath_settings + 'does not exist')
            
        # Check if folders exist
        #if not os.path.isdir(self.settingsDict['fp_images']):
        #    raise ValueError("Folderpath of image " + self.settingsDict['fp_images'] + ' does not exist')
        #if not os.path.isdir(self.settingsDict['folderpath_references']):
        #    raise ValueError("Folderpath of sgementations " + self.settingsDict['fp_seg_in'] + ' does not exist')

    def writeSettings(self, filepath_settings):
        """ Write settings into setting file

        :param filepath_settings: Filepath to settings file
        :type filepath_settings: str
        """

        # Initialize settings
        settingsDefault = {'method': 'xalabeler',
                           'fp_images': 'H:/cloud/cloud_data/Projects/CTLabeler/code/src/XALabeler/data/images',
                           'fp_pseudo': 'H:/cloud/cloud_data/Projects/CTLabeler/code/src/XALabeler/data/pseudo',
                           'fp_refine': 'H:/cloud/cloud_data/Projects/CTLabeler/code/src/XALabeler/data/refine',
                           'fp_mask': 'H:/cloud/cloud_data/Projects/CTLabeler/code/src/XALabeler/data/mask',
                           'fip_actionlist': 'H:/cloud/cloud_data/Projects/CTLabeler/code/src/XALabeler/data/sample/actionlist.json',
                           'fip_colors': 'H:/cloud/cloud_data/Projects/CTLabeler/code/src/XALabeler/data/XALabelerLUT.ctbl',
                           'foregroundOpacity': 0.3,
                           'classification': False,
                           'classification_multislice': False,
                           'show_roi': True,
                           'default_ignore': True}
                           
        print('Writing setting to ' + filepath_settings)
        with open(filepath_settings, 'w') as file:
            file.write(json.dumps(settingsDefault, indent=4))
        self.settingsDict = settingsDefault
        
    def checkSettings(self, settings):
        for key in settings.keys():
            value = settings[key]
            if isinstance(value, str):
                if "\\" in value:
                    raise ValueError("Backslash not allowed in settings file")
