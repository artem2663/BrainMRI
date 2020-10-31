from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Dropout, Input, Dense, Flatten, concatenate
from keras.models import Model
import tensorflow as tf
import numpy as np
import MRIDataset as md
import MRIDatasetV as mdv
import DB as db
import H5ToPBConverter as h5
from keras import backend as K


K.set_learning_phase(0)

database = db.DBManager() 

MRIs = md.MRIDataset("Datasets/**/**/", True, True)
train_images = np.array(MRIs.images)
train_labels = np.array(MRIs.labels)
ids = np.array(MRIs.ids)

dbRes = database.GetPatientsByCode(",".join(ids))
train_data = np.array(dbRes.loc[:, dbRes.columns != "code"])

vMRIs = mdv.MRIDatasetV("Validation/**/**/", True)
v_images = np.array(vMRIs.images)
v_labels = np.array(vMRIs.labels)
v_ids = np.array(vMRIs.ids)
v_dbRes = database.GetPatientsByCode(",".join(v_ids))
v_data = np.array(v_dbRes.loc[:, v_dbRes.columns != "code"])




convInput = Input(shape=(16, 256, 256, 1), name="convInput")
conv = Conv3D(64, kernel_size=(3, 3, 3), strides=(4,4,4), activation='relu')(convInput)
conv = MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2))(conv)
#conv = BatchNormalization(center=True, scale=True)(conv)
#conv = Dropout(0.5)(conv)
conv = Conv3D(128, kernel_size=(2, 3, 3), activation='relu')(conv)
conv = MaxPooling3D(pool_size=(1, 2, 2),strides=(2,2,2))(conv)
#conv = BatchNormalization(center=True, scale=True)(conv)
#conv = Dropout(0.5)(conv)
#conv = Conv3D(384, kernel_size=(1, 3, 3), activation='relu')(conv)
#conv = Conv3D(256, kernel_size=(1, 2, 2), activation='relu')(conv)
#conv = MaxPooling3D(pool_size=(1, 1, 1), strides=(2,2,2))(conv)


convOut = Flatten()(conv)
convOut = Dense(128, activation='relu')(convOut)


featInput = Input(shape=(2,), name="featInput")
feat = BatchNormalization()(featInput)
featOut = Dense(32, activation='relu')(feat)


concat = concatenate([convOut, featOut])
#concat = featOut

concat = Dense(32, activation='relu')(concat)
#concat = Dropout(0.5)(concat)

#concat = Dense(4096, activation='relu')(concat)
#concat = Dropout(0.5)(concat)

out = Dense(4, activation='softmax')(concat)


model = Model([convInput, featInput], out)
#model = Model(featInput, out)

print(model.summary())

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

res = model.fit([train_images, train_data], train_labels, epochs=30, verbose=1, validation_data=([v_images,v_data],v_labels))
#res = model.fit(train_data, train_labels, epochs=100, verbose=1)


'''
from matplotlib import pyplot

print(res.history['loss'])
pyplot.plot(res.history['loss'])
pyplot.plot(res.history['val_loss'])

pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.show()
'''



model.save("Models/CNNModel.h5")


converter = h5.Converter()
frozen_graph = converter.freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, 'Models/', 'CNNModel.pbtxt', as_text=True)
tf.train.write_graph(frozen_graph, 'Models/', 'CNNModel.pb', as_text=False)







