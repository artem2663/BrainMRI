import numpy as np
import MRIDataset as md
import MRIDatasetV as mdv


MRIs = md.MRIDataset("Datasets/**/**/", True, True)
train_images = np.array(MRIs.images)
train_labels = np.array(MRIs.labels)
ids = np.array(MRIs.ids)
np.save("LoadedData/train_images", train_images)
np.save("LoadedData/train_labels", train_labels)
np.save("LoadedData/train_ids", ids)


vMRIs = mdv.MRIDatasetV("Validation/**/**/", True)
v_images = np.array(vMRIs.images)
v_labels = np.array(vMRIs.labels)
v_ids = np.array(vMRIs.ids)
np.save("LoadedData/v_images", v_images)
np.save("LoadedData/v_labels", v_labels)
np.save("LoadedData/v_ids", v_ids)



