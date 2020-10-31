import glob
import numpy as np
from medpy.io import load
from skimage.transform import rescale, resize

from matplotlib import pyplot as plt

class MRIDataset:

    ids = []
    images = []
    labels = []


    def __init__(self, imageFolder, isTrain, flip=False):
        imageFolders = glob.glob(imageFolder, recursive=False)
        self.GetImages(imageFolders, isTrain)
        if flip == True:
            self.GetImages(imageFolders, isTrain, True)



    def GetImages(self, imageFolders, isTrain, flip=False):
        
        for imgFolder in imageFolders:

            if isTrain:
                label = int((imgFolder.split("\\")[1]).split("_")[0])
                self.labels.append(label)

                id = str((imgFolder.split("\\")[2]).split("_")[0])
                self.ids.append(id)


            files = glob.glob(imgFolder+"/*.dcm")

            dicomSet = []
            for img in files:
                image_data, image_header = load(img)

                image_data = image_data / np.max(image_data)
                image_data = resize(image_data[:,:,0],(256,256))
                if flip == True:
                    image_data = np.flip(image_data, 0)

                image_data = image_data[:,:, np.newaxis]

                #TODO: intensity normalization:
                #n = np.mean(image_data)
                #image_data *= 0.5 / n
                #image_data -= np.mean(image_data)
                #image_data /= np.std(image_data)
                
                #TODO: sharpen & contrast

                #plt.imshow(image_data[:,:,0],cmap="gray",vmax=1)
                #plt.show()

                orderNum = image_header.get_sitkimage().GetOrigin()[2]
                dicomSet.append([orderNum, image_data])

            
            dicomSet = sorted(dicomSet, key=lambda x: x[0])
            dicomSet = np.delete(dicomSet, 0, 1)
            
            idx = self.GetDistribution(len(dicomSet))
            dicomSet = np.take(dicomSet, idx)

            imgRes = []
            for img in dicomSet:
                imgRes.append(img)
            imgRes = np.asarray(imgRes)

            #plt.imshow(imgRes[0,:,:,0],cmap="gray",vmax=1)
            #plt.show()

            self.images.append(imgRes)
            



    def GetDistribution(self, n):
        idx = []
        x = round((n - 8) / 16, 6)

        i = x + 4
        while int(i) <= n - 4 and len(idx) < 16:
            idx.append(int(i))
            i += x 
        
        return idx
        
