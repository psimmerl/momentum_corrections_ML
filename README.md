# Proton Momentum Corrections for CLAS12


## Loss function minimum is in the wrong spot when doing simultaneous corrections

[MM2_Corrections_Error.pdf](MM2_Corrections_Error.pdf)

<!-- Loss function:
$$
J = \mathbb{E}\left[ \left(MM^2_{\pi_0} - MM^2_{pred}\right)^2\right]
$$

In practice I use: (equivalent to changing learning rate)
$$
J = \mathbb{E}\left[ \left(1 - \frac{MM^2_{pred}}{MM^2_{\pi_0}}\right)^2\right]
$$

$$
\pi_0 = beam + targ - e' - p' \\
MM^2 = (beam + targ - e' - p')^2
$$

$$
eE1' = \sqrt{(c1*ex')^2 + (c1*ey')^2 + (c1*ez')^2 + eM^2)}\\
pE2' = \sqrt{(c1*ex')^2 + (c1*ey')^2 + (c1*ez')^2 + eM^2)}\\
$$

$$
E = eE + pE - eE1'   - pE2'\\
x = ex + px - c1*ex' - c2*px'\\
y = ey + py - c1*ey' - c2*py'\\
z = ez + pz - c1*ez' - c2*pz'\\
$$


if c1, c2 = 0, 0:
$$
E = eE + pM - 0 - pM\\
x =  0 +  0 - 0 - 0\\
y =  0 +  0 - 0 - 0\\
z = eE +  0 - 0 - 0\\

MM^2 = ( eE )^2 - ( eE )^2 = 0\\
MM^2_{\pi_0} = 0.138^2 = 0.019044
$$

With the data:
* c1,c2=0,0 - $MM^2$ error: 0.0009 $GeV^2$ (0.03 $GeV$)
* c1,c2=1,1 - $MM^2$ error: 0.007 $GeV^2$ (0.08 $GeV$) -->

<!-- ---
# Archive

The neural network predicts $\Delta \log(\text{proton}_P)$, uses a custom layer to calculte $\text{MM}^2$, then simultaneously minimizes $\Delta \log(\text{proton}_P)$ and $\Delta \text{MM}^2$. (see the image at the bottom)
* Using $\log(\text{proton}_P)$ transforms the input and output distributions to pseudo-Gaussian.
* Bad exploding gradient problem -> use Adam/RMSprop with small learning_rate or SGD with clipnorm/clipval

```python
X = [ log(ele_P), ele_Theta, ele_Phi, log(bad_pro_P), bad_pro_Theta, bad_pro_Phi, log(bad_Q2), log(bad_t) ]
y = [ log(good_pro_P) - log(bad_pro_P), good_mm2 - bad_mm2 ]
```

## Performance:
* [Analysis.ipynb](analysis.ipynb)

## Feature Importances From a High Variance Random Forest:
> Using a random forest I got a better validation MSE without any regularization/tuning, might be good to test some other models too.
* 32.8% - bad_pro_Theta 
* 15.6% - log(bad_t) 
* 13.6% - bad_pro_P 
* 09.8% - log(q2) 
* 09.6% - ele_P 
* 06.7% - bad_pro_Phi 
* 06.4% - ele_Phi 
* 05.5% - ele_Theta 



## Model Summary:
```
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 8)]          0           []                               
                                                                                                  
 batch_normalization (BatchNorm  (None, 8)           32          ['input_1[0][0]']                
 alization)                                                                                       
                                                                                                  
 dense (Dense)                  (None, 417)          3753        ['batch_normalization[0][0]']    
                                                                                                  
 activation (Activation)        (None, 417)          0           ['dense[0][0]']                  
                                                                                                  
 batch_normalization_1 (BatchNo  (None, 417)         1668        ['activation[0][0]']             
 rmalization)                                                                                     
                                                                                                  
 dense_1 (Dense)                (None, 417)          174306      ['batch_normalization_1[0][0]']  
                                                                                                  
 activation_1 (Activation)      (None, 417)          0           ['dense_1[0][0]']                
                                                                                                  
 batch_normalization_2 (BatchNo  (None, 417)         1668        ['activation_1[0][0]']           
 rmalization)                                                                                     
                                                                                                  
 dense_2 (Dense)                (None, 417)          174306      ['batch_normalization_2[0][0]']  
                                                                                                  
 activation_2 (Activation)      (None, 417)          0           ['dense_2[0][0]']                
                                                                                                  
 batch_normalization_3 (BatchNo  (None, 417)         1668        ['activation_2[0][0]']           
 rmalization)                                                                                     
                                                                                                  
 dense_3 (Dense)                (None, 417)          174306      ['batch_normalization_3[0][0]']  
                                                                                                  
 activation_3 (Activation)      (None, 417)          0           ['dense_3[0][0]']                
                                                                                                  
 batch_normalization_4 (BatchNo  (None, 417)         1668        ['activation_3[0][0]']           
 rmalization)                                                                                     
                                                                                                  
 dense_4 (Dense)                (None, 417)          174306      ['batch_normalization_4[0][0]']  
                                                                                                  
 activation_4 (Activation)      (None, 417)          0           ['dense_4[0][0]']                
                                                                                                  
 batch_normalization_5 (BatchNo  (None, 417)         1668        ['activation_4[0][0]']           
 rmalization)                                                                                     
                                                                                                  
 dense_5 (Dense)                (None, 417)          174306      ['batch_normalization_5[0][0]']  
                                                                                                  
 activation_5 (Activation)      (None, 417)          0           ['dense_5[0][0]']                
                                                                                                  
 batch_normalization_6 (BatchNo  (None, 417)         1668        ['activation_5[0][0]']           
 rmalization)                                                                                     
                                                                                                  
 dense_6 (Dense)                (None, 417)          174306      ['batch_normalization_6[0][0]']  
                                                                                                  
 activation_6 (Activation)      (None, 417)          0           ['dense_6[0][0]']                
                                                                                                  
 batch_normalization_7 (BatchNo  (None, 417)         1668        ['activation_6[0][0]']           
 rmalization)                                                                                     
                                                                                                  
 dense_7 (Dense)                (None, 417)          174306      ['batch_normalization_7[0][0]']  
                                                                                                  
 activation_7 (Activation)      (None, 417)          0           ['dense_7[0][0]']                
                                                                                                  
 batch_normalization_8 (BatchNo  (None, 417)         1668        ['activation_7[0][0]']           
 rmalization)                                                                                     
                                                                                                  
 P (Dense)                      (None, 1)            418         ['batch_normalization_8[0][0]']  
                                                                                                  
 CalcMM2 (Lambda)               (None, 2)            0           ['input_1[0][0]',                
                                                                  'P[0][0]']                      
                                                                                                  
==================================================================================================
Total params: 1,237,689
Trainable params: 1,231,001
Non-trainable params: 6,688
__________________________________________________________________________________________________
```

![best model plot](models/best_model.png)
 -->
