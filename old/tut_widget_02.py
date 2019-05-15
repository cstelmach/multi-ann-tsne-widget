'''FastSNE Testwidget'''

import numpy as np
import os
import subprocess

from Orange.widgets.widget import OWWidget, Input
from Orange.widgets import gui

# import sys; sys.path.append('../')
# from Orange.widgets.bakk.fitsne.fast_tsne import fast_tsne


class Print(OWWidget):
    name = "Test FIt-SNE"
    description = "Print out a number"
    icon = "icons/print.svg"

    class Inputs:
        number = Input("Number", int)

    want_main_area = False

    def __init__(self):
        super().__init__()
        self.number = None

        #number = 10
        # number = open('/Applications/Orange3.app/Contents/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/Orange/widgets/bakk/data.txt', 'wb')
        # number_x = []
        # for x in range(0,1000):
        #     number_x.append(np.random.ranf(50))
        # number = fast_tsne(number_x, perplexity = 30)

        version_number = '1.1.0'
        nthreads = None
        number = subprocess.check_output(['/Applications/Orange3.app/Contents/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/Orange/widgets/bakk/fitsne/' +
                               '/bin/fast_tsne', version_number, 'data.dat', 'result.dat', '{}'.format(nthreads)])

        self.label = gui.widgetLabel(self.controlArea, "The number is {}".format(number))

    @Inputs.number
    def sef_number(self, number):
        """Set the input number."""

        self.number = number
        if self.number is None:
            self.label.setText("The number is: ??")
        else:
            self.label.setText("The number is {}".format(self.number))
