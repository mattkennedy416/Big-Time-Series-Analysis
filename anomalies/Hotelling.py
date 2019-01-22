


from external.banpei import banpei

def Hotelling(data, threshold):

    model = banpei.Hotelling()
    return model.detect(data, threshold)
