
from external.banpei import banpei

def SST(data, windowSize):

    model = banpei.SST(w=windowSize)
    return model.detect(data)


