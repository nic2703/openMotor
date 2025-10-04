from PyQt6.QtWidgets import QDialog, QApplication

from motorlib.properties import PropertyCollection, FloatProperty, StringProperty, EnumProperty
from ..converter import Exporter

from ..views.EngExporter_ui import Ui_EngExporterDialog

class EngSettings(PropertyCollection):
    def __init__(self):
        super().__init__()
        self.props['diameter'] = FloatProperty('Motor Diameter', 'm', 0, 1)
        self.props['length'] = FloatProperty('Motor Length', 'm', 0, 4)
        self.props['hardwareMass'] = FloatProperty('Hardware Mass', 'kg', 0, 1000)
        self.props['designation'] = StringProperty('Motor Designation')
        self.props['manufacturer'] = StringProperty('Motor Manufacturer')
        self.props['append'] = EnumProperty('Existing File', ['Append', 'Overwrite'])


class EngExportMenu(QDialog):
    def __init__(self, exporter):
        QDialog.__init__(self)
        self.ui = Ui_EngExporterDialog()
        self.ui.setupUi(self)

        self.setWindowIcon(QApplication.instance().icon)

        self.exporter = exporter

    def exec(self):
        newSettings = EngSettings()
        designation = self.exporter.manager.simRes.getDesignation()
        newSettings.setProperties({'designation': designation})
        self.ui.motorStats.setPreferences(self.exporter.manager.preferences)
        self.ui.motorStats.loadProperties(newSettings)
        if super().exec():
            return self.ui.motorStats.getProperties()
        return None


class EngExporter(Exporter):
    def __init__(self, manager):
        super().__init__(manager, 'ENG File',
            'Exports the results of a simulation in the RASP ENG format', {'.eng': 'RASP Files'}, False)
        self.menu = EngExportMenu(self)
        self.reqNotMet = "Must have run a simulation to export a .ENG file."

    def doConversion(self, path, config):
        mode = 'a' if config['append'] == 'Append' else 'w'
        with open(path, mode) as outFile:
            propMass = self.manager.simRes.getPropellantMass()
            contents = ' '.join([config['designation'],
                                 str(round(config['diameter'] * 1000, 6)),
                                 str(round(config['length'] * 1000, 6)),
                                 'P',
                                 str(round(propMass, 6)),
                                 str(round(propMass + config['hardwareMass'], 6)),
                                 config['manufacturer']
                                 ]) + '\n'

            timeData = self.manager.simRes.channels['time'].getData()
            forceData = self.manager.simRes.channels['force'].getData()
            # Add on a 0-thrust datapoint right after the burn to satisfy RAS Aero
            if forceData[-1] != 0:
                timeData.append(self.manager.simRes.getBurnTime() + 0.01)
                forceData.append(0)
            for time, force in zip(timeData, forceData):
                if time == 0: # Increase the first point so it isn't 0 thrust
                    force += 0.01
                contents += str(round(time, 4)) + ' ' + str(round(force, 4)) + '\n'

            contents += ';\n;\n'

            outFile.write(contents)

    def checkRequirements(self):
        return self.manager.simRes is not None
