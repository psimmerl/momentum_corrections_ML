"""
May need to do pretraining on p0 and e0 then switch over to MM2 training
Or use secondary losses to regularize submodels
Or decay the MM2E reg term e.g. 1000 to 0.1 over ~10 epochs
Or train them asynchronously(?)
"""
import ROOT
import numpy as np
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense, Dropout, Input, BatchNormalization,
                                     LayerNormalization, concatenate)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam, SGD
from tensorflow.keras.regularizers import L2


def convert_data(fname: str = 'data/raw/lvl2_eppi0.inb.mc.root',
                 ttree_name: str = 'h22',
                 seed: int = 42) -> tuple[np.ndarray]:
  rdf = ROOT.RDataFrame(ttree_name, fname)
  nevs = rdf.Count().GetValue()

  X_e_feats = ['eP', 'eTheta', 'ePhi', 'ePt', 'eEta']
  # X_e_feats += [f'esec{i}' for i in range(6)]
  X_p_feats = ['pP', 'pTheta', 'pPhi', 'pPt', 'pEta']
  y_feats = ['x', 'y', 'z']

  vals = ','.join(X_e_feats + X_p_feats)
  rdf = rdf.Define(
      'vals', '''
  TLorentzVector beam, targ, ele, pro;
  beam.SetXYZT(0,0,10.6041,10.6041);
  targ.SetXYZT(0,0,0,0.938272081);
  ele.SetXYZM(ex, ey, ez, 0.000510999);
  pro.SetXYZM(px, py, pz, 0.938272081);

  auto eE = ele.E(), eP = ele.P(), eTheta = ele.Theta(), ePhi = ele.Phi();
  auto pE = pro.E(), pP = pro.P(), pTheta = pro.Theta(), pPhi = pro.Phi();
  auto ePt = ele.Pt(), eEta = ele.Eta();
  auto pPt = pro.Pt(), pEta = pro.Eta();

  auto esec0=0., esec1=0., esec2=0., esec3=0., esec4=0., esec5=0.;
  if (esec == 1) esec0 = 1;
  if (esec == 2) esec1 = 1;
  if (esec == 3) esec2 = 1;
  if (esec == 4) esec3 = 1;
  if (esec == 5) esec4 = 1;
  if (esec == 6) esec5 = 1;

  auto Q2 = -(beam - ele).M2(), mt = -(targ - pro).M2();

  return vector<double>{''' + vals + '};')
  for i, val in enumerate(vals.split(',')):
    rdf = rdf.Define(val, f'vals[{i}]')

  # Should be doing this with a hash to guarantee reproducibility
  idxs = np.random.default_rng(seed=seed).permutation(nevs)
  X = np.stack([
      np.stack([rdf.AsNumpy([k])[k] for k in [ft for ft in X_e_feats]]).T,
      np.stack([rdf.AsNumpy([k])[k] for k in [ft for ft in X_p_feats]]).T
  ],
               axis=-1)[idxs, :, :]
  y = np.stack(
      [
          np.stack([rdf.AsNumpy(['e' + k])['e' + k] for k in y_feats]).T,
          np.stack([rdf.AsNumpy(['p' + k])['p' + k] for k in y_feats]).T
      ],  # + '0' 
      axis=-1)[idxs, :, :]

  # split data into 20% testing, 20% validation, and 60% training
  sidx = nevs // 5
  X_test, X_valid, X_train = X[:sidx], X[sidx:2 * sidx], X[2 * sidx:]
  y_test, y_valid, y_train = y[:sidx], y[sidx:2 * sidx], y[2 * sidx:]

  np.save('data/train_EPThetaPhi.npy', X_train)
  np.save('data/valid_EPThetaPhi.npy', X_valid)
  # np.save('test_EPThetaPhi.npy',X_test)

  np.save('data/train_XYZ.npy', y_train)
  np.save('data/valid_XYZ.npy', y_valid)
  # np.save('test_XYZ.npy',y_test)

  # eps = 1e-3
  # for i in range(8):
  #   xx = X_train[:, i%4, i//4]
  #   xmin, xmax = np.min(xx), np.max(xx)
  #   if i%4 != 3:
  #     if np.mean(xx) < np.median(xx):
  #       X_train[:, i%4, i//4] = np.log(xmax - X_train[:, i%4, i//4] + eps)
  #       X_valid[:, i%4, i//4] = np.log(xmax - X_valid[:, i%4, i//4] + eps)
  #       X_test[:, i%4, i//4] = np.log(xmax - X_test[:, i%4, i//4] + eps)
  #     else:
  #       X_train[:, i%4, i//4] = np.log(X_train[:, i%4, i//4] - xmin + eps)
  #       X_valid[:, i%4, i//4] = np.log(X_valid[:, i%4, i//4] - xmin + eps)
  #       X_test[:, i%4, i//4] = np.log(X_test[:, i%4, i//4] - xmin + eps)

  print([np.any(np.isnan(xx)) for xx in (X_train, X_valid, X_test)])
  return X_train, X_valid, X_test, y_train, y_valid, y_test


class MissingMassSquaredError(keras.losses.Loss):
  """Missing Mass Squared Error"""

  def __init__(self,
               beam_E: float = 10.6041,
               reg: float = False,
               name: str = 'MM2E') -> None:
    super().__init__(name=name)
    self.PION_MASS = 0.1349768  # GeV
    self.PROTON_MASS = 0.9382721  # GeV
    self.ELECTRON_MASS = 0.0005110  # GeV

    self.METRIC = tf.constant([1., -1., -1., -1.])
    self.BEAM = tf.constant([beam_E, 0, 0, beam_E])
    self.TARGET = tf.constant([self.PROTON_MASS, 0, 0, 0])
    self.BT = self.BEAM + self.TARGET

    self.reg = reg

  def call(self,
           y_true,
           y_pred,
           sample_weight=None,
           print_vals=False) -> tf.Tensor:

    y_true = tf.cast(y_true, tf.float32) # throws error if I naively call method
    y_pred = tf.cast(y_pred, tf.float32) # throws error if I naively call method
    # y_pred = tf.exp(y_pred)

    # Assume the correction factor is never lower than 0.5
    e_factor = y_pred[:, 0, None] / 100 + 0.5
    p_factor = y_pred[:, 1, None] / 100 + 0.5

    ele0 = y_true[:, :, 0]
    pro0 = y_true[:, :, 1]

    eleP2_pred = e_factor**2 * tf.reduce_sum(ele0**2, axis=1, keepdims=True)
    proP2_pred = p_factor**2 * tf.reduce_sum(pro0**2, axis=1, keepdims=True)

    ele = tf.concat(
        [(self.ELECTRON_MASS**2 + eleP2_pred)**(1 / 2), e_factor * ele0],
        axis=1)
    pro = tf.concat(
        [(self.PROTON_MASS**2 + proP2_pred)**(1 / 2), p_factor * pro0], axis=1)

    MM2_pred = tf.reduce_sum(self.METRIC * (self.BT - ele - pro)**2, axis=1)
    if print_vals:
      print('preds', e_factor[:3].numpy(), p_factor[:3].numpy())
      print('e0,p0', ele0[:3].numpy(), pro0[:3].numpy())
      print('P2_pred', eleP2_pred[:3].numpy(), proP2_pred[:3].numpy())
      print('e,p', ele[:3].numpy(), pro[:3].numpy())
      print('mm2', MM2_pred[:10].numpy())

    if self.reg:
      return tf.reduce_mean((1 - MM2_pred / self.PION_MASS**2)**2) + \
                            self.reg * tf.math.reduce_mean((1 - y_pred)**2)
    # return tf.reduce_mean((1 - MM2_pred / self.PION_MASS**2)**2)
    return tf.reduce_mean(( self.PION_MASS**2 - MM2_pred )**2)


def build_submodel(
    n_inputs=4) -> tuple[keras.layers.Layer, keras.layers.Layer]:
  input = keras.layers.Input(shape=n_inputs)  # E, |p|, theta, phi
  x = BatchNormalization()(input)
  # x = LayerNormalization()(input)
  # x = Normalize()(input)
  for ilay in range(4):
    x = Dense(
        units=256,
        activation='relu',
        kernel_initializer='he_normal',
    )(x)  #kernel_regularizer=L2()
    x = BatchNormalization()(x)
    # x = LayerNormalization()(x)
    # x = Dropout(0.4)(x)
  # using softplus to prevent going negative
  output = Dense(units=1, activation='softplus')(x)
  # output = Dense(units=1, activation='linear')(x)
  return input, output


def build_model(n_inputs=4, MM2E_reg=False) -> tuple[keras.models.Model]:
  ele_in, ele_out = build_submodel(n_inputs=n_inputs)
  pro_in, pro_out = build_submodel(n_inputs=n_inputs)
  concat_outputs = concatenate([ele_out, pro_out])

  ele_model = keras.Model(inputs=[ele_in], outputs=[ele_out], name='Electron')
  pro_model = keras.Model(inputs=[pro_in], outputs=[pro_out], name='Proton')
  full_model = keras.Model(inputs=[ele_in, pro_in],
                           outputs=[concat_outputs],
                           name='Full')

  # opt = SGD(learning_rate=1e-4, nesterov=True)
  opt = Adam()
  full_model.compile(loss=MissingMassSquaredError(reg=MM2E_reg),
                     optimizer=opt)  #, jit_compile=True)  #, run_eagerly=True)
  return full_model, ele_model, pro_model


def MM2E_to_MM(mm2e):
  PION_MASS = 0.1349768  # GeV
  return np.sqrt(mm2e)  # / PION_MASS**2


if __name__ == '__main__':
  tf.random.set_seed(42)
  np.random.seed(42)

  X_train, X_valid, X_test, y_train, y_valid, y_test = convert_data()
  model, ele_model, pro_model = build_model(n_inputs=X_train.shape[1],
                                            MM2E_reg=False)

  mm2_loss = MissingMassSquaredError()

  zvec = (np.zeros((len(y_train), 1)) - 0.5) * 100
  ovec = (np.ones((len(y_train), 1)) - 0.5) * 100
  loss00 = mm2_loss.call(y_train, np.c_[zvec,
                                        zvec]).numpy()  #, print_vals=True
  loss01 = mm2_loss.call(y_train, np.c_[zvec,
                                        ovec]).numpy()  #, print_vals=True
  loss10 = mm2_loss.call(y_train, np.c_[ovec,
                                        zvec]).numpy()  #, print_vals=True
  loss11 = mm2_loss.call(y_train, np.c_[ovec,
                                        ovec]).numpy()  #, print_vals=True
  print(f"0,0 - Loss: {loss00:.3f}, MM2E: {MM2E_to_MM(loss00):.6f}")
  print(f"0,1 - Loss: {loss01:.3f}, MM2E: {MM2E_to_MM(loss01):.6f}")
  print(f"1,0 - Loss: {loss10:.3f}, MM2E: {MM2E_to_MM(loss10):.6f}")
  print(f"1,1 - Loss: {loss11:.3f}, MM2E: {MM2E_to_MM(loss11):.6f}")

  callbacks = [
      keras.callbacks.EarlyStopping(monitor='val_loss',
                                    patience=10,
                                    restore_best_weights=True),
  ]
  X_trn2 = [X_train[:, :, 0], X_train[:, :, 1]]
  X_val2 = [X_valid[:, :, 0], X_valid[:, :, 1]]

  # print(X_trn2, [y_train[:, :, 0], y_train[:, :, 1]])

  history = model.fit(
      X_trn2,
      y_train,
      validation_data=(X_val2, y_valid),
      epochs=100,  #50,
      batch_size=32,
      callbacks=callbacks)

  trn_pred = model(X_trn2, training=False).numpy()
  val_pred = model(X_val2, training=False).numpy()
  trn_loss = mm2_loss.call(y_train, trn_pred, print_vals=True).numpy()
  val_loss = mm2_loss.call(y_valid, val_pred, print_vals=True).numpy()

  print('ave,std =', np.mean(trn_pred, axis=0), np.std(trn_pred, axis=0))
  trn_pred = trn_pred / 100 + 0.5
  val_pred = val_pred / 100 + 0.5
  print('ave,std =', np.mean(trn_pred, axis=0), np.std(trn_pred, axis=0))

  np.save('data/train_pred.npy', trn_pred)
  np.save('data/valid_pred.npy', val_pred)

  print(f"Trn Loss: {trn_loss}, RMM2E: {MM2E_to_MM(trn_loss):.6f}")
  print(f"Val Loss: {val_loss}, RMM2E: {MM2E_to_MM(val_loss):.6f}")
  print(np.round(val_pred[:10].T, 3))
  # print(np.round(np.exp(model(X_val2[:10]).numpy().T) + 0.5,3))
