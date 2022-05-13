import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Lambda, BatchNormalization, Dropout, LeakyReLU, Activation
from tensorflow.keras.models import Sequential
import keras_tuner as kt

X_train = np.load('data/X_train_all_feats.npy')
y_train = np.load('data/y_train_all_feats.npy')

X_valid = np.load('data/X_valid_all_feats.npy')
y_valid = np.load('data/y_valid_all_feats.npy')

def CalcMM2(x):
  # Cartesian
  # mean_X  = tf.constant([1.04684507e-03, 3.99276990e-03, 5.54965746e+00, -1.35728314e-03, -3.17695929e-03, 9.60693938e-01])
  # scale_X = tf.constant([0.97338996, 0.97730009, 1.3635807, 0.40069781, 0.40065214, 0.49844655])

  # Spherical
  # mean_X  = tf.constant([ 5.72991007,  0.2579039,  -0.09691791,  1.09272128,  0.48494906, -0.01878646])
  # scale_X = tf.constant([1.31483893, 0.08475765, 1.82449035, 0.54629188, 0.13847994, 1.82014073])

  # Spherical with log(Q2) and log(t)
  # mean_X  = tf.constant([5.729910068550966, 0.25790390434770494, -0.0969179068165942, 1.0927212793110743, 0.48494905542094124, -0.018786463892064623, 1.263540776502531, -0.2643839382949719, ])
  # scale_X = tf.constant([1.3148389266636, 0.08475764952525691, 1.8244903543417919, 0.5462918841848049, 0.13847994287641613, 1.8201407301860681, 0.3892811506389084, 0.7268083601619448, ])

  # Spherical with log(p0), log(pP), log(eP), log(Q2), and log(t)
  mean_X  = tf.constant([1.712782369548623, 0.25790390434770494, -0.0969179068165942, -0.013550632960287997, 0.48494905542094124, -0.018786463892064623, 1.263540776502531, -0.2643839382949719, ])
  scale_X = tf.constant([0.27440008795120685, 0.08475764952525691, 1.8244903543417919, 0.4399088483504666, 0.13847994287641613, 1.8201407301860681, 0.3892811506389084, 0.7268083601619448, ])
  mean_y  = tf.constant([0.02577764878163872, 0.025417884420961278, ])
  scale_y = tf.constant([0.04114841268106677, 0.09791073484206517, ])

  emass, pmass, beamE = 0.000510999, 0.938272081, 10.6041

  in_fixed = x[0]*scale_X + mean_X

  # exyz, pxyz = in_fixed[:,:3], in_fixed[:,3:6]
  eptf, pptf = in_fixed[:,:3], in_fixed[:,3:6]

  p = x[1]*scale_y[0] + mean_y[0]
  p = tf.exp(p + pptf[:,0:1])
  # p = tf.exp(x[1])

  eptf = tf.concat([tf.exp(eptf[:,0:1]), eptf[:,1:]], axis=1)
  pptf = tf.concat([tf.exp(pptf[:,0:1]), pptf[:,1:]], axis=1)

  exyz = tf.stack([eptf[:,0]*tf.cos(eptf[:,2])*tf.sin(eptf[:,1]), 
                    eptf[:,0]*tf.sin(eptf[:,2])*tf.sin(eptf[:,1]), 
                    eptf[:,0]*tf.cos(eptf[:,1])], axis=1)
  pxyz = tf.stack([pptf[:,0]*tf.cos(pptf[:,2])*tf.sin(pptf[:,1]), 
                    pptf[:,0]*tf.sin(pptf[:,2])*tf.sin(pptf[:,1]), 
                    pptf[:,0]*tf.cos(pptf[:,1])], axis=1)

  beam = tf.constant([0., 0., beamE, np.sqrt(beamE**2 + emass**2)])
  targ = tf.constant([0., 0., 0., pmass])

  eleE = tf.sqrt(emass**2 + eptf[:,0:1]**2)
  ele  = tf.concat([exyz, eleE], axis=1)

  proE_X = tf.sqrt(pmass**2 + pptf[:,0:1]**2)
  pro_X  = tf.concat([pxyz, proE_X], axis=1)
  mm2_X = tf.reduce_sum((beam + targ - ele - pro_X)**2 * [-1, -1, -1, 1], axis=1, keepdims=True)

  pp0 = p / pptf[:,0:1]

  proE = tf.sqrt(pmass**2 + pp0**2 * pptf[:,0:1]**2)
  pro  = tf.concat([pp0*pxyz, proE], axis=1)
  mm2 = tf.reduce_sum((beam + targ - ele - pro)**2 * [-1, -1, -1, 1], axis=1, keepdims=True)
  
  # return tf.concat([x[1], mm2], axis=1)

  dmm2 = (mm2 - mm2_X)
  dmm2 = (dmm2 - mean_y[1]) / scale_y[1]
  return tf.concat([x[1], dmm2], axis=1)

def build_model(hp):
  opt    = hp.Choice("opt", ['Adam', 'RMSprop'])#, 'SGD'])
  act    = hp.Choice("act", ['elu', 'relu', 'LeakyReLU'])
  # cv     = hp.Int("clip", min_value=1,  max_value=10)
  layers = hp.Int("layers", min_value=2,    max_value=8)
  units  = hp.Int("units",  min_value=64,   max_value=1024)#,  sampling='log')
  lr     = hp.Float("lr",   min_value=1e-7, max_value=1e-2, sampling='log')
  DO     = hp.Float("DO",   min_value=0.0,  max_value=0.5,  step=0.1, default=0.0)
  ki     = 'he_normal'

  input_ = Input(shape=X_train.shape[1:])
  if DO > 0: x = Dropout(DO)(input_)
  else: x = BatchNormalization()(input_)
  
  for i in range(layers):
    x = Dense(units=units, kernel_initializer=ki)(x)# if i else input_)

    if act == 'LeakyReLU': x = LeakyReLU(alpha=0.2)(x)
    else: x = Activation(act)(x)

    if DO > 0: x = Dropout(DO)(x)
    else: x = BatchNormalization()(x)
  
  x = Dense(units=1, activation='linear', name='P')(x)
  output = Lambda(CalcMM2, name='CalcMM2')([input_, x])

  model = keras.Model(inputs=[input_], outputs=[output])

  if opt == 'Adam':
    optimizer = keras.optimizers.Adam(learning_rate=lr)#, clipvalue=cv)
  elif opt == 'RMSprop':
    optimizer = keras.optimizers.RMSprop(learning_rate=lr)#, rho=0.9)#, clipvalue=cv)
  elif opt == 'SGD':
    optimizer = keras.optimizers.SGD(learning_rate=lr, nesterov=True)#, clipvalue=1.0)

  model.compile(loss='mse', optimizer=optimizer, jit_compile=True)

  # model.summary()
  # keras.utils.plot_model(model, 'mm2_model.png')#, show_shapes=True)
  return model


if __name__ == '__main__':
  # from joblib import load
  # scalerX = load('data/scaler_X.joblib')
  # mean_X, scale_X = scalerX.mean_, scalerX.scale_
  # print(mean_X, scale_X)

  # scalery = load('data/scaler_y.joblib')
  # mean_y, scale_y = scalery.mean_, scalery.scale_
  # print(mean_y, scale_y)

  callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(f'models/model.h5', save_best_only=True),
    keras.callbacks.TensorBoard('logs/tensorboard/')
  ]

  tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=500, 
  # tuner = kt.BayesianOptimization(build_model, objective='val_loss', max_trials=250, 
                          overwrite=True, directory='logs', project_name='diy_space_heater', seed=42)
  
  try:
    tuner.search(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_valid, y_valid), callbacks=callbacks, verbose=2)
  except:
    print("Keyboard Interrupt!")

  tuner.results_summary()
  best_model = tuner.get_best_models(num_models=1)[0]

  print("\nEvaluating Best Model:")
  best_model.evaluate(X_train, y_train)
  best_model.evaluate(X_valid, y_valid)

  keras.utils.plot_model(best_model, 'models/best_model.png', show_shapes=True, show_layer_activations=True)
  best_model.save('models/best_model.h5')
