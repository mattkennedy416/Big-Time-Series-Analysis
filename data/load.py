

import pandas as pd
import numpy as np
import os
from glob import glob

class load():
    # lets just make this a class so we can get specific suggestions rather than having to enter strings

    def __init__(self):
        self.subdir = 'data'


    def _findPath(self, path):

        possibilities = [path,
                         self.subdir + os.sep + path,
                         '../' + self.subdir + os.sep + path,
                         '../../' + self.subdir + os.sep + path]

        for possibility in possibilities:
            if os.path.isfile(possibility):
                return possibility
            elif os.path.isdir(possibility):
                return possibility

        raise FileNotFoundError('Unable to find file ' + str(path))


    def auslan(self):
        rootDir = self._findPath('auslan')

        paths = glob(rootDir + '/*')
        out = {}

        headers = ['x1','y1','z1','roll1','pitch1','yaw1','thumb1','forefinger1','middlefinger1','ringfinger1','littlefinger1',
                   'x2', 'y2', 'z2', 'roll2', 'pitch2', 'yaw2', 'thumb2', 'forefinger2', 'middlefinger2', 'ringfinger2','littlefinger2']


        for path in paths:
            label = os.path.split(path)[1].replace('.tsd', '')

            out[label] = pd.read_csv(path, delimiter='\t', names=headers)

        return out






    def amazon(self):
        filename = 'AMZN.txt'
        filepath = self._findPath(filename)

        frame = pd.read_csv(filepath)

        del frame['<ticker>']
        return frame.values


    def s_and_p500(self):
        filename ='GSPC.csv'
        filepath = self._findPath(filename)

        frame = pd.read_csv(filepath)

        del frame['Date']
        return frame.values


    def TICC(self):
        filename = 'TICC_example_data.txt'
        filepath = self._findPath(filename)

        frame = pd.read_csv(filepath, header=None)
        return frame.values


    def ekg(self):
        # from: https://github.com/mrahtz/sanger-machine-learning-workshop
        import struct

        filename = 'ekg.dat'
        filepath = self._findPath(filename)

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
