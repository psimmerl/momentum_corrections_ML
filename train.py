"""
May need to do pretraining on p0 and e0 then switch over to MM2 training
Or use secondary losses to regularize submodels
Or decay the MM2E reg term e.g. 1000 to 0.1 over ~10 epochs
Or train them asynchronously(?)
"""
import ROOT
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, BatchNormalization, concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from tensorflow.keras.regularizers import L2


def convert_data(fname: str = 'data/raw/lvl2_eppi0.inb.mc.root',
                 ttree_name: str = 'h22',
                 seed: int = 42) -> tuple[np.ndarray]:
  rdf = ROOT.RDataFrame(ttree_name, fname)
  nevs = rdf.Count().GetValue()

  X_feats = ['E', 'P', 'Theta', 'Phi']
  y_feats = ['x', 'y', 'z']

  vals = 'eE,eP,eTheta,ePhi,pE,pP,pTheta,pPhi'
  rdf = rdf.Define(
      'vals', '''
  TLorentzVector ele(ex, ey, ez, 0.000510999), pro(px, py, pz, 0.938272081);
  auto eE = ele.E(), eP = ele.P(), eTheta = ele.Theta(), ePhi = ele.Phi();
  auto pE = pro.E(), pP = pro.P(), pTheta = pro.Theta(), pPhi = pro.Phi();
  return vector<double>{''' + vals + '};')
  for i, val in enumerate(vals.split(',')):
    rdf = rdf.Define(val, f'vals[{i}]')

  # Should be doing this with a hash to guarantee reproducibility
  idxs = np.random.default_rng(seed=seed).permutation(nevs)
  X = np.stack([
      np.stack([rdf.AsNumpy([k])[k] for k in ['e' + ft for ft in X_feats]]).T,
      np.stack([rdf.AsNumpy([k])[k] for k in ['p' + ft for ft in X_feats]]).T
  ],
               axis=-1)[idxs, :, :]
  y = np.stack([
      np.stack([rdf.AsNumpy([k])[k] for k in ['e' + ft for ft in y_feats]]).T,
      np.stack([rdf.AsNumpy([k])[k] for k in ['p' + ft + '0' for ft in y_feats]]).T
  ],
               axis=-1)[idxs, :, :]

  # split data into 20% testing, 20% validation, and 60% training
  split_idx = nevs // 5
  X_test, X_valid, X_train = X[:split_idx, :, :], X[split_idx:2 * split_idx, :, :], X[2 * split_idx:, :, :]
  y_test, y_valid, y_train = y[:split_idx, :, :], y[split_idx:2 * split_idx, :, :], y[2 * split_idx:, :, :]

  return X_train, X_valid, X_test, y_train, y_valid, y_test


class MissingMassSquaredError(keras.losses.Loss):
  """Missing Mass Squared Error"""
  def __init__(self, beam_E: float = 10.6041, reg: float = False, name: str = 'MM2E') -> None:
    super().__init__(name=name)
    self.PION_MASS = 0.1349768  # GeV
    self.PROTON_MASS = 0.9382721  # GeV
    self.ELECTRON_MASS = 0.0005110  # GeV

    self.METRIC = tf.constant([1., -1., -1., -1.])
    self.BEAM = tf.constant([beam_E, 0, 0, (beam_E**2 - self.ELECTRON_MASS**2)**(1 / 2)])
    self.TARGET = tf.constant([self.PROTON_MASS, 0, 0, 0])
    self.BT = self.BEAM + self.TARGET

    self.reg = reg

  def call(self, y_true, y_pred, sample_weight=None) -> tf.Tensor:
    # print(y_pred[:3], y_true[:3])
    eleP2_pred = y_pred[:, 0, None]**2 * tf.reduce_sum(y_true[:, :, 0]**2, axis=-1, keepdims=True)
    proP2_pred = y_pred[:, 1, None]**2 * tf.reduce_sum(y_true[:, :, 1]**2, axis=-1, keepdims=True)
    # print(eleP2_pred[:3], proP2_pred[:3])

    ele = tf.concat([(self.ELECTRON_MASS**2 + eleP2_pred)**(1 / 2), y_pred[:, 0, None] * y_true[:, :, 0]], axis=1)
    pro = tf.concat([(self.PROTON_MASS**2 + proP2_pred)**(1 / 2), y_pred[:, 1, None] * y_true[:, :, 1]], axis=1)
    # print(ele[:3], pro[:3])

    MM2_pred = tf.reduce_sum(self.METRIC * (self.BT - ele - pro)**2, axis=-1, keepdims=True)
    # print(MM2_pred[:3])

    if self.reg:
      return tf.reduce_mean((1 - MM2_pred / self.PION_MASS**2)**2) + \
                            self.reg * tf.math.reduce_mean(tf.square(1 - y_pred))
    return tf.reduce_mean((1 - MM2_pred / self.PION_MASS**2)**2)


def build_submodel() -> tuple[keras.layers.Layer, keras.layers.Layer]:
  input = keras.layers.Input(shape=4)  # E, |p|, theta, phi
  x = BatchNormalization()(input)
  for ilay in range(4):
    x = Dense(units=128, activation='elu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
  # using softplus to prevent going negative
  output = Dense(units=1, activation='softplus')(x)
  return input, output


def build_model(MM2E_reg=False) -> tuple[keras.models.Model]:
  ele_in, ele_out = build_submodel()
  pro_in, pro_out = build_submodel()
  concat_outputs = concatenate([ele_out, pro_out])

  ele_model = keras.Model(inputs=[ele_in], outputs=[ele_out], name='Electron')
  pro_model = keras.Model(inputs=[pro_in], outputs=[pro_out], name='Proton')
  full_model = keras.Model(inputs=[ele_in, pro_in], outputs=[concat_outputs], name='Full')

  opt = Adam()  #clipnorm=1.0)
  full_model.compile(loss=MissingMassSquaredError(reg=MM2E_reg), optimizer=opt)#, jit_compile=True)  #, run_eagerly=True)
  return full_model, ele_model, pro_model


if __name__ == '__main__':
  tf.random.set_seed(42)
  np.random.seed(42)

  X_train, X_valid, X_test, y_train, y_valid, y_test = convert_data()
  model, ele_model, pro_model = build_model(MM2E_reg=False)

  mm2_loss = MissingMassSquaredError()
  print(f"0 - Naive train MM2E: {mm2_loss.call(y_train, np.zeros((len(y_train), 2))).numpy()}")
  print(f"0 - Naive valid MM2E: {mm2_loss.call(y_valid, np.zeros((len(y_valid), 2))).numpy()}")
  print(f"1 - Naive train MM2E: {mm2_loss.call(y_train, np.ones((len(y_train), 2))).numpy()}")
  print(f"1 - Naive valid MM2E: {mm2_loss.call(y_valid, np.ones((len(y_valid), 2))).numpy()}")

  callbacks = [
      keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
  ]
  history = model.fit([X_train[:, :, 0], X_train[:, :, 1]],
                      y_train,
                      validation_data=([X_valid[:, :, 0], X_valid[:, :, 1]], y_valid),
                      epochs=50,
                      batch_size=32,
                      callbacks=callbacks)

  model.evaluate([X_train[:, :, 0], X_train[:, :, 1]], y_train)
  model.evaluate([X_valid[:, :, 0], X_valid[:, :, 1]], y_valid)

  print(f"Train MM2E for train: {mm2_loss.call(y_train, model([X_train[:,:,0], X_train[:,:,1]])).numpy()}")
  print(f"Valid MM2E for valid: {mm2_loss.call(y_valid, model([X_valid[:,:,0], X_valid[:,:,1]])).numpy()}")
  print(model([X_valid[:, :, 0], X_valid[:, :, 1]])[:10])
