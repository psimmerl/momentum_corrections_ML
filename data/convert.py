import ROOT
import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

rdf = ROOT.RDataFrame("h22", "raw/lvl2_eppi0.inb.mc.root")
nevs = rdf.Count().GetValue() 
print(nevs)

vals = "mm2,mm20,p0,eP,etheta,ephi,pP,ptheta,pphi,q2,t,dp,dmm2"
rdf = rdf.Define("vals", """
double emass = 0.000510999, promass = 0.938272081;
TLorentzVector beam, targ, ele, pro, pro0;
beam.SetXYZM(0, 0, 10.6041, emass);
targ.SetXYZM(0, 0, 0, promass);
ele.SetXYZM(ex, ey, ez, emass);
pro.SetXYZM(px, py, pz, promass);
pro0.SetXYZM(px0, py0, pz0, promass);

auto q2 = log(-(beam - ele).M2());
auto t = log(-(targ - pro).M2());

auto mm20 = (beam + targ - ele - pro0).M2();
auto mm2 = (beam + targ - ele - pro).M2();
auto p0 = pro0.P();

auto eP = ele.P();
auto etheta = ele.Theta();
auto ephi = ele.Phi();

auto pP = pro.P();
auto ptheta = pro.Theta();
auto pphi = pro.Phi();

eP = log(eP);
p0 = log(p0);
pP = log(pP);

auto dp = (p0 - pP);
auto dmm2 = (mm20 - mm2);

return vector<double>{"""+vals+"};")
for i, val in enumerate(vals.split(',')):
  rdf = rdf.Define(val, f"vals[{i}]")

X_feats = ["eP", "etheta", "ephi", "pP", "ptheta", "pphi", "q2", "t"]
# X_feats = ["ex", "ey", "ez", "px", "py", "pz"]#["ihel", "ex", "ey", "ez", "px", "py", "pz", "g1x", "g1y", "g1z", "g2x", "g2y", "g2z", "idet", "esec", "run", "status"]
# y_feats = ["p0", "mm20"]#["px0", "py0", "pz0"]
y_feats = ["dp", "dmm2"]#["px0", "py0", "pz0"]
X_dict = rdf.AsNumpy(X_feats)
y_dict = rdf.AsNumpy(y_feats)

X_all = np.zeros((nevs, len(X_feats)))
y_all = np.zeros((nevs, len(y_feats)))

for i, feat in enumerate(X_feats): X_all[:,i] = X_dict[feat]
for i, feat in enumerate(y_feats): y_all[:,i] = y_dict[feat]

# 60% train, 20% valid, 20% test
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.4, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_valid = scaler_X.transform(X_valid)
X_test = scaler_X.transform(X_test)
dump(scaler_X, 'scaler_X.joblib')
mean_X, scale_X = scaler_X.mean_, scaler_X.scale_

for dd in [mean_X, scale_X]:
  print('[', end='')
  for d in dd:
    print(d, end=', ')
  print(']')

scaler_y = StandardScaler() # technically not needed?
y_train = scaler_y.fit_transform(y_train)
y_valid = scaler_y.transform(y_valid)
y_test = scaler_y.transform(y_test)
dump(scaler_y, 'scaler_y.joblib')
mean_y, scale_y = scaler_y.mean_, scaler_y.scale_

for dd in [mean_y, scale_y]:
  print('[', end='')
  for d in dd:
    print(d, end=', ')
  print(']')

np.save('X_train_all_feats.npy', X_train)
np.save('X_valid_all_feats.npy', X_valid)
np.save('X_test_all_feats.npy',  X_test)

np.save('y_train_all_feats.npy', y_train)
np.save('y_valid_all_feats.npy', y_valid)
np.save('y_test_all_feats.npy',  y_test)

