import tensorflow as tf
import numpy as np
import MRIDataset as md


model = tf.keras.models.load_model("Models/CNNModel.h5")



mri = md.MRIDataset("TestData", False)
test_images = np.array(mri.images)

res = model.predict([test_images,np.array([[54, 1]])])   #astr
#res = model.predict(np.array([[57, 1]]))   #astr


print(res)
