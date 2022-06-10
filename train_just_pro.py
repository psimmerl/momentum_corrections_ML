'''
May need to do pretraining on p0 and e0 then switch over to MM2 training
Or use secondary losses to regularize submodels
Or decay the MM2E reg term e.g. 1000 to 0.1 over ~10 epochs
Or train them asynchronously(?)
'''
import ROOT
import numpy as np
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense, Input, Dropout, BatchNormalization, LayerNormalization, concatenate, LeakyReLU)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from tensorflow.keras.regularizers import L2


def convert_data(fname: str = 'data/raw/lvl2_eppi0.inb.mc.root',
                 ttree_name: str = 'h22',
                 seed: int = 42) -> tuple[np.ndarray]:
  rdf = ROOT.RDataFrame(ttree_name, fname)
  nevs = rdf.Count().GetValue()

  X_feats = ['P', 'Theta', 'Phi', 'Pt', 'Eta']
  y_feats = ['x', 'y', 'z']

  vals = 'eE,eP,eTheta,ePhi,ePt,eEta,pE,pP,pTheta,pPhi,pPt,pEta'
  rdf = rdf.Define(
      'vals', '''
  TLorentzVector ele, pro;
  ele.SetXYZM(ex, ey, ez, 0.000510999);
  pro.SetXYZM(px, py, pz, 0.938272081);

  auto eE = ele.E(), eP = ele.P(), eTheta = ele.Theta(), ePhi = ele.Phi();
  auto pE = pro.E(), pP = pro.P(), pTheta = pro.Theta(), pPhi = pro.Phi();
  auto ePt = ele.Pt(), eEta = ele.Eta();
  auto pPt = pro.Pt(), pEta = pro.Eta();

  return vector<double>{''' + vals + '};')
  for i, val in enumerate(vals.split(',')):
    rdf = rdf.Define(val, f'vals[{i}]')

  # Should be doing this with a hash to guarantee reproducibility
  idxs = np.random.default_rng(seed=seed).permutation(nevs)
  X = np.stack([rdf.AsNumpy([k])[k] for k in ['p' + ft for ft in X_feats]]).T[idxs, :]
  y = np.stack([np.stack([rdf.AsNumpy([k])[k] for k in ['e' + ft for ft in y_feats]]).T,
                np.stack([rdf.AsNumpy([k])[k] for k in ['p' + ft for ft in y_feats]]).T], # + '0'
                axis=-1)[idxs, :, :]

  # split data into 20% testing, 20% validation, and 60% training
  split_idx = nevs // 5
  X_test, X_valid, X_train = X[:split_idx], X[split_idx:2*split_idx], X[2*split_idx:]
  y_test, y_valid, y_train = y[:split_idx], y[split_idx:2*split_idx], y[2*split_idx:]

  return X_train, X_valid, X_test, y_train, y_valid, y_test


class MissingMassSquaredError(keras.losses.Loss):
  '''Missing Mass Squared Error'''
  def __init__(self, beam_E: float = 10.6041, reg: float = False, name: str = 'MM2E') -> None:
    super().__init__(name=name)
    self.PION_MASS = 0.1349768  # GeV
    self.PROTON_MASS = 0.9382721  # GeV
    self.ELECTRON_MASS = 0.0005110  # GeV

    self.METRIC = tf.constant([1., -1., -1., -1.])
    self.BEAM = tf.constant([beam_E, 0, 0, beam_E])
    self.TARGET = tf.constant([self.PROTON_MASS, 0, 0, 0])
    self.BT = self.BEAM + self.TARGET

    self.reg = reg

  def call(self, y_true, y_pred, sample_weight=None, print_vals=False) -> tf.Tensor:
    y_pred = tf.cast(y_pred, y_true.dtype)
    # print(y_pred[:3], y_true[:3])

    p_factor = y_pred
    e_factor = tf.ones_like(p_factor)

    ele0 = y_true[:, :, 0]
    pro0 = y_true[:, :, 1]

    eleP2_pred = e_factor**2 * tf.reduce_sum(ele0**2, axis=1, keepdims=True)
    proP2_pred = p_factor**2 * tf.reduce_sum(pro0**2, axis=1, keepdims=True)
    # print(eleP2_pred[:3], proP2_pred[:3])

    ele = tf.concat([(self.ELECTRON_MASS**2 + eleP2_pred)**(1/2), e_factor * ele0], axis=1)
    pro = tf.concat([(self.PROTON_MASS**2 + proP2_pred)**(1/2), p_factor * pro0], axis=1)
    # print(ele[:3], pro[:3])

    MM2_pred = tf.reduce_sum(self.METRIC * (self.BT - ele - pro)**2, axis=1)
    if print_vals:
      print(MM2_pred[:10].numpy().flatten())

    if self.reg:
      return tf.reduce_mean((1 - MM2_pred / self.PION_MASS**2)**2) + \
                            self.reg * tf.math.reduce_mean((1 - y_pred)**2)
    return tf.reduce_mean((self.PION_MASS**2 - MM2_pred)**2)
    # return tf.reduce_mean((1 - MM2_pred / self.PION_MASS**2)**2)


def build_submodel(n_inputs=4) -> tuple[keras.layers.Layer, keras.layers.Layer]:
  input = keras.layers.Input(shape=n_inputs)  # E, |p|, theta, phi
  x = LayerNormalization()(input)
  for ilay in range(4):
    x = Dense(units=256, 
              activation='relu',#LeakyReLU(alpha=0.2),
              kernel_initializer='he_normal')(x)
    # x = BatchNormalization()(x)
    # x = LayerNormalization()(x)
    # x = Dropout(0.4)(x)
  # using softplus to prevent going negative
  output = Dense(units=1, activation='softplus')(x)
  # output = Dense(units=1, activation='linear')(x)
  return input, output


def build_model(n_inputs=4, MM2E_reg=False) -> tuple[keras.models.Model]:
  pro_in, pro_out = build_submodel(n_inputs=n_inputs)

  pro_model = keras.Model(inputs=[pro_in], outputs=[pro_out], name='Proton')

  opt = Adam()  #clipnorm=1.0)
  pro_model.compile(loss=MissingMassSquaredError(reg=MM2E_reg), optimizer=opt)#, jit_compile=True)#, run_eagerly=True)
  return pro_model

def MM2E_to_MME(mm2e):
  PION_MASS = 0.1349768 # GeV
  return np.sqrt(mm2e)

if __name__ == '__main__':
  tf.random.set_seed(42)
  np.random.seed(42)

  X_train, X_valid, X_test, y_train, y_valid, y_test = convert_data()
  model = build_model(n_inputs=X_train.shape[1], MM2E_reg=False)

  mm2_loss = MissingMassSquaredError()

  zvec, ovec = np.zeros((len(y_train), 1)), np.ones((len(y_train), 1))
  loss0 = mm2_loss.call(y_train, zvec).numpy()#, print_vals=True 
  loss1 = mm2_loss.call(y_train, ovec).numpy()#, print_vals=True
  print(f'0 - Loss: {loss0:.3f}, MM2E: {MM2E_to_MME(loss0):.6f}')
  print(f'1 - Loss: {loss1:.3f}, MM2E: {MM2E_to_MME(loss1):.6f}')

  callbacks = [
      keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
  ]
  # for epoch in range(10):
  history = model.fit(X_train,
                      y_train,
                      validation_data=(X_valid, y_valid),
                      epochs=100,#50,
                      batch_size=32,
                      callbacks=callbacks)

    # model.evaluate(X_train, y_train)
    # model.evaluate(X_valid, y_valid)

    # print(f'Train MM2E for train: {mm2_loss.call(y_train, model(X_train)).numpy()}')
    # print(f'Valid MM2E for valid: {mm2_loss.call(y_valid, model(X_valid)).numpy()}')
  trn_loss = mm2_loss.call(y_train, model(X_train, training=False),
                print_vals=True).numpy()
  val_loss = mm2_loss.call(y_valid, model(X_valid, training=False),
                print_vals=True).numpy()
  print(f'Trn Loss: {trn_loss:.3f}, RMM2E: {MM2E_to_MME(trn_loss):.6f}')
  print(f'Val Loss: {val_loss:.3f}, RMM2E: {MM2E_to_MME(val_loss):.6f}')
  # print(np.round(model(X_valid[:10]).numpy().flatten(),3))
  print(np.round(model(X_valid[:10]).numpy().flatten(),3))
