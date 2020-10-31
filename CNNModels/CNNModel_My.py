from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Dropout, Input, Dense, Flatten, concatenate
from keras.models import Model
import tensorflow as tf
import numpy as np
import DB as db
import H5ToPBConverter as h5
from keras import backend as K


K.set_learning_phase(0)

database = db.DBManager() 

train_images = np.load("LoadedData/train_images.npy") 
train_labels = np.load("LoadedData/train_labels.npy")
ids = np.load("LoadedData/train_ids.npy") 
dbRes = database.GetPatientsByCode(",".join(ids))
train_data = np.array(dbRes.loc[:, dbRes.columns != "code"])

v_images = np.load("LoadedData/v_images.npy") 
v_labels = np.load("LoadedData/v_labels.npy") 
v_ids = np.load("LoadedData/v_ids.npy") 
v_dbRes = database.GetPatientsByCode(",".join(v_ids))
v_data = np.array(v_dbRes.loc[:, v_dbRes.columns != "code"])

#################



convInput = Input(shape=(16, 256, 256, 1), name="convInput")
conv = Conv3D(64, kernel_size=(3, 5, 5), strides=(4,4,4), activation='relu')(convInput)
conv = MaxPooling3D(pool_size=(3, 3, 3), strides=(2,2,2))(conv)

conv = Conv3D(128, kernel_size=(1, 5, 5), activation='relu')(conv)
conv = MaxPooling3D(pool_size=(1, 2, 2),strides=(2,2,2))(conv)

#conv = Conv3D(384, kernel_size=(1, 3, 3), activation='relu')(conv)
#conv = MaxPooling3D(pool_size=(1, 2, 2),strides=(2,2,2))(conv)

#conv = Conv3D(256, kernel_size=(1, 3, 3), activation='relu')(conv)
#conv = MaxPooling3D(pool_size=(1, 2, 2),strides=(2,2,2))(conv)

convOut = Flatten()(conv)
convOut = Dense(128, activation='relu')(convOut)


featInput = Input(shape=(2,), name="featInput")
feat = BatchNormalization()(featInput)
featOut = Dense(64, activation='relu')(feat)


concat = concatenate([convOut, featOut])
concat = Dense(64, activation='relu')(concat)
out = Dense(4, activation='softmax')(concat)


model = Model([convInput, featInput], out)


print(model.summary())

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit([train_images, train_data], train_labels, epochs=40, verbose=1, validation_data=([v_images,v_data],v_labels))



model.save("Models/CNNModel_Alt.h5")


converter = h5.Converter()
frozen_graph = converter.freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, 'Models/', 'CNNModel_Alt.pbtxt', as_text=True)
tf.train.write_graph(frozen_graph, 'Models/', 'CNNModel_Alt.pb', as_text=False)







