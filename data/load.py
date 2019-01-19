

import pandas as pd
import numpy as np
import os

class load():
    # lets just make this a class so we can get specific suggestions rather than having to enter strings

    def __init__(self):
        self.subdir = 'data'


    def _findFile(self, filename):

        possibilities = [filename,
                         self.subdir+os.sep+filename,
                         '../'+self.subdir+os.sep+filename]

        for possibility in possibilities:
            if os.path.isfile(possibility):
                return possibility

        raise FileNotFoundError('Unable to find file '+str(filename))


    def amazon(self):
        filename = 'AMZN.txt'
        filepath = self._findFile(filename)

        frame = pd.read_csv(filepath)

        del frame['<ticker>']
        return frame.values


    def s_and_p500(self):
        filename ='GSPC.csv'
        filepath = self._findFile(filename)

        frame = pd.read_csv(filepath)

        del frame['Date']
        return frame.values

    def TICC(self):
        filename = 'TICC_example_data.txt'
        filepath = self._findFile(filename)

        frame = pd.read_csv(filepath, header=None)
        return frame.values












if __name__ == '__main__':

    data = load().amazon()

    print(data)
