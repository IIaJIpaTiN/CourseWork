import os
from PIL import Image
import numpy as np

directory = 'data/face/'
saveDir = 'data/set/'
setSize = 20
width = 100

catalogs = os.listdir(directory)
curCatalog = 0

for catalog in catalogs:

    files = os.listdir(directory + catalog)
    images = list(filter(lambda x: x.endswith('.jpg'), files))
    curCatalog += 1
    os.mkdir(saveDir + str(curCatalog))

    for i in range(setSize):
        ind = np.random.randint(len(images))

        #Изменение ширины с сохнанением пропорции
        img = Image.open(directory + catalog + '/' + images[ind]).convert('L')
        percent = (width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(percent)))
        img = img.resize((width, hsize), Image.ANTIALIAS)

        background = Image.new('L', (width, width))
        background.paste(img)
        background.save(saveDir + str(curCatalog) + '/%d.jpg' % (i + 1))

        del images[ind]