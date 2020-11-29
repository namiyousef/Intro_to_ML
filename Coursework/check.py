"""
Note: this is code that will be used to check the performance of the models.
"""

import os.path
from keras.models import load_model
import numpy as np
import pandas as pd
from keras.utils import to_categorical

# user defined parameters (change these as necessary):
# put your name here:
givenname = 'Yousef'
familyname = 'Nami'
# set which dataset to use:
dataset = 2

filebase = familyname.lower()+'-'+givenname.lower()+'-'+np.str(dataset)
# load in student model
model = load_model(filebase+'.h5', compile=False)

if os.path.exists(filebase+'.txt'):
    print(filebase+'.txt exists - loading in scaling parameters')
    scaleArray = np.loadtxt(filebase+'.txt')
else:
    print(filebase + '.txt doesn\'t exist - assuming no scaling')
    scaleArray = np.array([np.zeros([6, ]), np.ones([6, ])])

# load in the data provided to the students
df = pd.read_csv('dataset' + np.str(dataset) + '.csv')
print(df.head())
Lt = np.array(df['Arm length (m)'][:])
Wt = np.array(df['Ball weight (kg)'][:])
Rt = np.array(df['Ball radius (mm)'][:])
Tt = np.array(df['Air temperature (deg C)'][:])
Et = np.array(df['Spring constant (N per m)'][:])
Dt = np.array(df['Device weight (kg)'][:])
Ot = np.array(df['Target hit'][:])
XtUnscaled = np.column_stack([Lt, Wt, Rt, Tt, Et, Dt])

# use values to scale validation data in XvUnscaled (whose shape is [number_of_validations,6])
Xt = (XtUnscaled-scaleArray[0, :])/scaleArray[1, :]

Yt = to_categorical(Ot)
# run the data through the model
Yt_predict = model.predict(Xt)

# output a summary of the model if you wish
print(model.summary())


number_correct = 0
for i in range(len(Yt)):
    if np.round(Yt[i, 0]) == np.round(Yt_predict[i, 0]):
        number_correct += 1

fraction_correct = 1.0 * number_correct / len(Yt_predict)
print(fraction_correct)

if fraction_correct < 0.6:
    print('Warning: very poor performance on provided data; likely error')
