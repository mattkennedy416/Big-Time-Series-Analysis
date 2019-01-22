

from external.banpei import banpei
from data.load import load
import matplotlib.pyplot as plt

data = load().ekg()[0:1000]


model = banpei.SST(w=200)
results = model.detect(data)


model2 = banpei.Hotelling()


plt.plot(data)
plt.plot(100*results)
plt.show()
















