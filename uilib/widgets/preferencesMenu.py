from PyQt6.QtWidgets import QDialog, QApplication
from PyQt6.QtCore import pyqtSignal

from uilib.views.Preferences_ui import Ui_PreferencesDialog


class PreferencesMenu(QDialog):

    preferencesApplied = pyqtSignal(dict)

    def __init__(self):
        QDialog.__init__(self)

        self.ui = Ui_PreferencesDialog()
        self.ui.setupUi(self)

        self.setWindowIcon(QApplication.instance().icon)

        self.ui.buttonBox.accepted.connect(self.apply)
        self.ui.buttonBox.rejected.connect(self.cancel)

    def load(self, pref):
        self.ui.settingsEditorGeneral.setPreferences(pref)
        self.ui.settingsEditorGeneral.loadProperties(pref.general)
        self.ui.settingsEditorUnits.loadProperties(pref.units)

    def apply(self):
        self.preferencesApplied.emit({'general': self.ui.settingsEditorGeneral.getProperties(),
                                      'units': self.ui.settingsEditorUnits.getProperties()})
        self.hide()

    def cancel(self):
        self.hide()
