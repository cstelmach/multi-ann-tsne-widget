import sys
#sys.path.insert(0, "/Library/Frameworks/Python.framework/Versions/3.7/lib/python37.zip")
#sys.path.insert(0, "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7")
#sys.path.insert(0, "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/lib-dynload")
#sys.path.insert(0, "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages")
#sys.path.insert(0, "/Users/make/Library/Python/3.7/lib/python/site-packages")
#sys.path.insert(0, "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages")
#import nearpy

#from Orange.widgets.bakk.additional_files.algorithms import nearpy

from Orange.widgets.widget import OWWidget, Output
from Orange.widgets import gui


class IntNumber(OWWidget):
    # Widget's name as displayed in the canvas
    name = "Integer Number"
    # Short widget description
    description = "Lets the user input a number"

    # An icon resource file path for this widget
    # (a path relative to the module where this widget is defined)
    icon = "icons/number.svg"

    # Widget's outputs; here, a single output named "Number", of type int
    class Outputs:
        number = Output("Number", int)

    # Basic (convenience) GUI definition:
    # a simple 'single column' GUI layout
    want_main_area = False
    # with a fixed non resizable geometry.
    resizing_enabled = False


    neighbor_method_index = ["nndescent", "balltree", "annoy", "hnsw", "bruteforce", "nearpy", "exact", "approx"]
    neighbor_method = ContextSetting("nndescent")

    #number = OWWidget.settings.Setting(42)
    number = sys.path

    def __init__(self):
        super().__init__()
        gui.lineEdit(self.controlArea, self, "number", "Enter a number",
                     box="Number",
                     callback=self.number_changed,
                     valueType=int, validator=None)

        self.method_combo = gui.comboBox(
            self.controlArea, self, "neighbor_method", orientation=Qt.Horizontal,
            label="ANN:", items=["nndescent", "balltree", "annoy", "hnsw", "bruteforce", "nearpy", "exact", "approx"], #[i for i in self.neighbor_method_index],
            sendSelectedValue=True,  callback=self._invalidate_affinities)

        self.number_changed()

    def number_changed(self):
        # Send the entered number on "Number" output
        self.Outputs.number.send(self.number)
