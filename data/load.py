

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
                         '../'+self.subdir+os.sep+filename,
                         '../../'+self.subdir+os.sep+filename]

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


    def ekg(self):
        # from: https://github.com/mrahtz/sanger-machine-learning-workshop
        import struct

        filename = 'ekg.dat'
        filepath = self._findFile(filename)

        with open(filepath, 'rb') as input_file:
            data_raw = input_file.read()
        n_bytes = len(data_raw)
        n_shorts = n_bytes / 2
        # data is stored as 16-bit samples, little-endian
        # '<': little-endian
        # 'h': short
        unpack_string = '<%dh' % n_shorts
        # sklearn seems to throw up if data not in float format
        data_shorts = np.array(struct.unpack(unpack_string, data_raw)).astype(float)
        return data_shorts






if __name__ == '__main__':

    data = load().ekg()

    print(data)
