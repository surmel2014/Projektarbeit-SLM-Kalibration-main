#Allgemeines
import numpy as np
import tensorflow as tf
from tensorflow import keras
#tf.autograph.set_verbosity(3, True)
from scipy.stats import norm
from PIL import Image
import datetime
#AO-Tools: Zernike Layer
import aotools

# Layer_combined_v6
# Rechteckige SLM-Dimension
# Erweiterung durch Gitterstruktur des SLMs

def to_image_plane(array:np.ndarray):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(array)))

def to_holo_plane(array:np.ndarray):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(array)))



#@keras_export("keras.constraints.NonNeg64")
#benutzt im Gauss-Layer
class NonNeg64(tf.keras.constraints.Constraint):
    """
    Constrains the weights to be non-negative.
    Also available via the shortcut function `tf.keras.constraints.non_neg`.
    """
    def __call__(self, w):
        return w * tf.cast(tf.greater_equal(w, 0.0), tf.float64)


'''
Gauss-Profil
'''
def show_gauss(size, weights): 
    num_gauss = weights[0].size
    gauss_single = 0
    gauss = 0
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    for i in range(num_gauss):
        gauss_single = np.exp(-weights[3][i]*(xx-weights[1][i])**2 - weights[3][i]*(yy-weights[2][i])**2)
        gauss = gauss + weights[0][i]*gauss_single
    return gauss



'''
Funktion fuer Polynome: variabler Grad, mit Verschiebung x-x0, coeff = [x0, x1, x2,...]
'''
def show_phase_to_voltage(inputs, coeff, shift):
    degree = len(coeff)
    voltage = 0
    for i in range(degree):
        voltage = voltage + coeff[i]*(inputs - shift[i])**(i+1)
    #voltage = np.clip(voltage, 0, 1)
    return voltage





class GaussLayer(tf.keras.layers.Layer):
    '''
    guide for custom Layer:
    https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing
    '''
    def __init__(self, num_gauss, **kwargs):
        self.num_gauss = num_gauss
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.amp_wg = self.add_weight(shape= (self.num_gauss,), initializer = "random_normal", trainable=True, constraint = NonNeg64(), dtype = tf.float64, name = 'amp_wg')
        self.amp_wg[0].assign(1)
        self.coord_x = self.add_weight(shape= (self.num_gauss,), initializer = "random_normal", trainable=True, dtype = tf.float64, name = 'coord_x')
        self.coord_y = self.add_weight(shape= (self.num_gauss,), initializer = "random_normal", trainable=True, dtype = tf.float64, name = 'coord_y')
        self.width = self.add_weight(shape= (self.num_gauss,), initializer = "random_normal", trainable=True, constraint = NonNeg64(), dtype = tf.float64, name = 'width_gauss')
        linvector_x = tf.linspace(0, 1, input_shape[-1])
        linvector_y = tf.linspace(0, 1, input_shape[-2])
        self.xx, self.yy = tf.meshgrid(linvector_x, linvector_y)

    def create_gauss(self, inputs, amp_wg, coord_x, coord_y, width):
        size_x = inputs.get_shape()[-2]#Pixel x
        size_y = inputs.get_shape()[-1]#Pixel y
        gauss_sum = tf.zeros([size_x, size_y],dtype=tf.float64)
        for i in range(self.num_gauss):
            gauss = tf.math.exp(-1*width[i]*(self.xx-coord_x[i])**2 -1*width[i]*(self.yy-coord_y[i])**2)
            gauss_sum = gauss_sum + gauss*amp_wg[i]
        gauss_sum_complex = tf.cast( tf.complex(gauss_sum, tf.cast(0,tf.float64)), tf.complex64)
        complex_field = gauss_sum_complex*tf.complex(tf.math.cos(inputs), tf.math.sin(inputs))
        return complex_field

    def call(self, inputs):
        return self.create_gauss(inputs, self.amp_wg, self.coord_x, self.coord_y, self.width)



# 膎derung in zernike array 25.01.2024
# Die Phasen der Zernike Polynome  in der build-funktion erstellen (aotools.functions.zernikeArray) und als self-varibale vorhalten. Bei call die phasen mit self.polynom multiplizieren und somit gewichten (wesentlich schneller)
class ZernikeLayer(tf.keras.layers.Layer):
    '''
    Custom Layer for Zernike Fields
    aotools for zernike field
    guide for custom Layer:
    https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing
    '''
    def __init__(self, num_zernike = 5, **kwargs):
        self.num_zernike = num_zernike
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.polynom = self.add_weight(shape=(self.num_zernike,) ,initializer="random_normal", trainable=True, name = 'polynom_zernike')
        size_x = input_shape[-1] #1920
        size_y = input_shape[-2] #1080
        zernike_size = int(size_x* 1.5) #2880 (diameter)
        #cut_x = int((zernike_size-size)/2)
        #cut_y = int((zernike_size-size)/2)
        cut_x = 480
        cut_y = 900
        self.zernike_phase = aotools.zernikeArray(self.num_zernike +1,zernike_size)[1:,cut_y:-cut_y,cut_x:-cut_x]

    def call(self, inputs):
        weighted_zernikes = tf.math.reduce_sum(tf.expand_dims(tf.expand_dims(self.polynom, -1), -1) * self.zernike_phase, 0)
        return weighted_zernikes + inputs



class PhasetoVoltageLayer(tf.keras.layers.Layer):
    '''
    26.02.24 aenderung der Funktion und weglassen des Terms i=0
    guide for custom Layer:
    https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing
    
    '''
    def __init__(self, num_grad, **kwargs):
        self.num_grad = num_grad
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_grad": self.num_grad,
        })
        return config

    def build(self,input_shape):
        
        self.coeff = self.add_weight(shape=(self.num_grad, ), initializer="random_normal", trainable=True, name = 'coeff')
        self.coeff[0].assign(1)
        self.coeff[1].assign(0.1)
        self.coeff[3].assign(0.1)
        self.shift = self.add_weight(shape=(self.num_grad, ), initializer="random_normal", trainable=True, name = 'shift')#+0.01 #initializer="random_normal",
        #self.shift[0].assign(0)

    def phase_to_voltage(self, inputs, coeff, shift):
        #voltage = coeff[0]*tf.math.pow(inputs, 0)+coeff[1]*tf.math.pow(inputs, 1)+coeff[2]*tf.math.pow(inputs, 2)+coeff[3]*inputs+coeff[3]+coeff[4]*inputs+coeff[4]
        #voltage = tf.clip_by_value(voltage, 0, 1)
        voltage = 0
        for i in range(self.num_grad):
            voltage = voltage + coeff[i]*(inputs - shift[i])**(i+1)
        #voltage = tf.clip_by_value(voltage, 0, 1)
        return voltage
    
    def call(self, inputs):
        return self.phase_to_voltage(inputs, self.coeff, self.shift)



class gauss_initializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        
        return tf.constant(tf.expand_dims(tf.expand_dims([[.1, .1, .1], [.1, .9, .1], [.1, .1, .1]],-1),-1))


class ScaleLayer(tf.keras.layers.Layer):
    '''
    scales the field with a constant
    guide for custom Layer:
    https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self,input_shape):
        
        self.scale = self.add_weight(shape=(1,), initializer="Ones", trainable=True, name = 'scale')#, constraint=NonNeg64())
        self.scale[0].assign(0.5)

    def call(self, inputs):
        return inputs*self.scale

class cPhaseLayer(tf.keras.layers.Layer):
    '''
    creates constant phase values
    guide for custom Layer:
    https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self,input_shape):
        self.phase = tf.zeros(shape = (1,*input_shape[1:]))
        
    def call(self, inputs):
        return self.phase

class SincLayer(tf.keras.layers.Layer):
    '''
    creates a sinc-squared pattern and multiply with input
    guide for custom Layer:
    https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self,input_shape):
        num = input_shape[-1]
        lin_vec = tf.linspace(-1,1, num)
        meshgrid = tf.meshgrid(lin_vec, lin_vec)
        self.grid = tf.cast(tf.math.sqrt(meshgrid[0]**2 + meshgrid[1]**2), dtype = tf.float64)
        #self.width = self.add_weight(shape=(1,), initializer="Ones", trainable=True, dtype = tf.float64, constraint=NonNeg64(), name = 'width_sinc')
        
        self.width = self.add_weight(shape=(1,), initializer="Ones", trainable=True, dtype = tf.float64, 
                                     constraint=keras.constraints.MinMaxNorm(min_value=0.75, max_value=1.0, rate=1.0, axis=0), 
                                     name = 'width_sinc')
        
        self.width[0].assign(0.2) #small number -> broad sinc

    def create_sinc(self):
        return tf.cast(tf.experimental.numpy.sinc(self.grid*self.width)**2, tf.float32)
            
    def call(self, inputs):
        return inputs*self.create_sinc()



def build_model(input_size=(1, 1080,1920),name:str=None):
    '''
    input: Phase values from [-pi;pi]
    output: amplitude at far field
    '''

    gauss_layer =GaussLayer(num_gauss = 3)
    zernike_layer = ZernikeLayer(num_zernike=10)
    
    input_phase = tf.keras.Input(input_size)
    phasetovoltage = PhasetoVoltageLayer(num_grad=10)(input_phase)
    '''
    phasetovoltage = tf.keras.layers.Conv2D(filters=1,
                                            kernel_size = (3,3),
                                            kernel_initializer=gauss_initializer(),
                                            kernel_constraint=tf.keras.constraints.NonNeg(),
                                            padding ='same',
                                            use_bias = False,
                                            data_format='channels_first')(phasetovoltage)
    '''
    phasetovoltage = tf.multiply(2*np.pi, phasetovoltage) #12.12.23 input von [0,1] zu [0,2pi]
    aberrated_phase = zernike_layer(phasetovoltage)
    field_at_slm = gauss_layer(aberrated_phase)

    padded_field_slm = tf.keras.layers.ZeroPadding2D(padding = (3780,3360),data_format = 'channels_first')(field_at_slm)
    intensity_far_slm = tf.square(tf.abs(tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(padded_field_slm)))) /2000)

    
    #----Gitter----
    grating_phase_initial = cPhaseLayer()(input_phase)
    grating_phase_zernike = zernike_layer(grating_phase_initial)
    field_grating = gauss_layer(grating_phase_zernike)
    
    padded_field_grating = tf.keras.layers.ZeroPadding2D(padding = (3780,3360),data_format = 'channels_first')(field_grating)
    intensity_far_grating = tf.square(tf.abs(tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(padded_field_grating)))) /2000)
    intensity_far_grating = ScaleLayer()(intensity_far_grating)
    
    #----beide Intensitaeten addiert
    out_intensity = intensity_far_slm + intensity_far_grating # Gitter ausgeblendet
    #out_intensity = intensity_far_slm
    
    '''
    out_intensity =  tf.keras.layers.Conv2D(filters = 1, 
                                            kernel_size = (4,4), 
                                            strides = (1,1),
                                            padding = 'same', 
                                            use_bias = False, 
                                            kernel_initializer = tf.keras.initializers.Constant(value=0.0625), 
                                            data_format = "channels_first", 
                                            trainable = False)(out_intensity)
    '''    
    out_intensity = SincLayer()(out_intensity)
    
    out_intensity = tf.clip_by_value(out_intensity, 0,1)
    #cut_dim_nn = 1920 # muss unbedingt auch unten ge鋘dert werden
    out = out_intensity[:,:,4320:6240,4320:6240] #cut only region with signal
    model = tf.keras.Model(inputs=input_phase, outputs=[out, out_intensity],name=name)
    return model


if __name__ == '__main__':

    '''
    Training mit mehreren Bildern 31.01.2024
    '''

    image_names = ['0703_aeffle_pferdle_crop_step2.png',
                   '0703_ito_crop_step2.png',
                   '0703_ito_inverted_crop_step2.png',
                   '0703_spie_crop_step2.png']

    phase_holo_names = ['0703_aeffle_pferdle_phase_step2.jpg',
                  '0703_ito_phase_step2.jpg',
                  '0703_ito_inverted_phase_step2.jpg',
                  '0703_spie_phase_step2.jpg']

    training_images = []
    training_phases = []
    for i in range(len(image_names)):
        img = Image.open(image_names[i]).convert(mode='L')
        img_arr = np.asarray(img)/255
        img_arr = np.flip(img_arr, -2)
        img_arr = np.expand_dims(img_arr,0)
        training_images.append(img_arr)
        
        phs = Image.open(phase_holo_names[i]).convert(mode='L')
        phs_arr = np.asarray(phs)/255
        #phs_arr = np.flip(phs_arr,1)
        phs_arr = np.expand_dims(phs_arr,0)
        training_phases.append(phs_arr)

    training_images = np.asarray(training_images)
    training_phases = np.asarray(training_phases)


    '''
    Neuronales Netz startet hier
    '''

    #Tensorboard
    #http://localhost:6006/#timeseries
    #cd Pfad waehlen
    #Eingabe: 'tensorboard --logdir .'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    model_combined = build_model()
    model_combined.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model_combined.compile(optimizer=tf.keras.optimizers.Adam(), loss=['mse', None], metrics=['mse'])
    model_combined.fit(training_phases, [training_images, np.zeros((4,1,3840,3840))], batch_size=2, epochs=1000, callbacks=[tensorboard_callback])


    prediction = model_combined.predict(np.expand_dims(training_phases[2,:,:,:],0))

    diff = prediction[0] - training_images[2,:,:,:]

    '''
    Weights
    '''
    weights_phasetovoltage = model_combined.layers[1].get_weights()
    x = np.linspace(0, 1, 100)
    graph_predicted = show_phase_to_voltage(x, weights_phasetovoltage[0], weights_phasetovoltage[1])
    
    #weights_conv = model_combined.layers[2].get_weights()
    weights_zernicke = model_combined.layers[4].get_weights()
    weights_gauss = model_combined.layers[5].get_weights()
    weights_scale = model_combined.layers[6].get_weights()

    size = training_phases[1].shape[2]
    gauss_prediction=show_gauss(size, weights_gauss)


    '''
    Model.save
    '''
    '''
    model_combined.save('saved_model '+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    '''