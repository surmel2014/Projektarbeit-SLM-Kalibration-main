import numpy as np
import tensorflow as tf



def prop_to_far_field_sgd1(phase, size_slm_pad_x, size_slm_pad_y, size_slm_x) -> tf.Tensor:
    phase = phase*2*np.pi
    slm_plane = tf.complex(tf.math.cos(phase), tf.math.sin(phase))
    slm_plane = tf.pad(slm_plane,((size_slm_pad_y,size_slm_pad_y),(size_slm_pad_x,size_slm_pad_x)))
    return tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(slm_plane))) / size_slm_x

def loss_sgd1(slm_phase, size_slm_padding_x, size_slm_padding_y, size_slm_x, target_intensity, size_image, energy_scaling):
    diff_sum = 0
    far_field = prop_to_far_field_sgd1(slm_phase, size_slm_pad_x=size_slm_padding_x, size_slm_pad_y=size_slm_padding_y, size_slm_x=size_slm_x)
    far_field_intensity = tf.square(tf.abs(far_field))
    diff = target_intensity - far_field_intensity[size_image:size_image+1920,size_image:size_image+1920] * energy_scaling

    diff_mean = tf.math.reduce_mean(tf.square(diff))
    return diff_mean


def SGD_initial_phase(target_intensity, size_slm_y, size_slm_x, size_image, size_slm_padding_y, size_slm_padding_x, loss_list):
    
    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.01,
                decay_steps=100,
                decay_rate=0.9,
                staircase=True)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    initial_phase = np.random.rand(size_slm_y, size_slm_x)
    #initial_phase = np.zeros((size_slm_y, size_slm_x))
    slm_phase = tf.Variable(initial_phase, trainable=True, name='phase',constraint=lambda z: tf.math.floormod(z, 1))

    target_intensity = tf.Variable(target_intensity*255, trainable = False, name='target') # SGD Step_1 auf 0-255 normiert, laeuft dann besser
    energy_scaling = tf.Variable((1),trainable=True,dtype=tf.float64,name='scale1',constraint=lambda z: tf.abs(z))

    for j in range(150):
        with tf.GradientTape() as tp:
            loss_fn = loss_sgd1(slm_phase, size_slm_padding_x, size_slm_padding_y, size_slm_x, target_intensity, size_image, energy_scaling)
        gradients = tp.gradient(loss_fn, [slm_phase, energy_scaling])
        opt.apply_gradients(zip(gradients, [slm_phase, energy_scaling]))
        
        current_loss = loss_fn.numpy()
        loss_list.append(current_loss)
        if (j+1) % 10 == 0:
            print(j+1)
            print(current_loss)
    intensity_far_field = prop_to_far_field_sgd1(slm_phase,size_slm_pad_x=size_slm_padding_x, size_slm_pad_y=size_slm_padding_y, size_slm_x=size_slm_x).numpy()

    computed_phase = np.array(slm_phase)
    return computed_phase, intensity_far_field


def loss_sgd_model(model, slm_phase, target_intensity, size_image, energy_scaling):
    diff_sum = 0

    far_field_intensity = model(slm_phase)[0][0,0,:,:]
    diff = target_intensity - far_field_intensity #* energy_scaling
    
    diff_mean = tf.math.reduce_mean(tf.square(diff))

    return diff_mean


def SGD_with_model(model, initial_phase, image_intensity, size_image, loss_list):
            
    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=100,
        decay_rate=0.9,
        staircase=True)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

    initial_phase_expand = np.expand_dims(np.expand_dims(initial_phase,0),0)

    slm_phase = tf.Variable(initial_phase_expand, trainable=True, name='phase',constraint=lambda z: tf.math.floormod(z, 1), dtype = tf.float32)
    target_intensity = tf.Variable(image_intensity, trainable = False, name='target', dtype = tf.float32)
    energy_scaling = tf.Variable((1),trainable=False,dtype=tf.float32,name='scale1',constraint=lambda z: tf.abs(z))

    for j in range(100):
        with tf.GradientTape() as tp:
            loss_fn = loss_sgd_model(model, slm_phase, target_intensity, size_image, energy_scaling)
        gradients = tp.gradient(loss_fn, [slm_phase])#, energy_scaling])
        opt.apply_gradients(zip(gradients, [slm_phase]))#, energy_scaling]))
        
        current_loss = loss_fn.numpy()
        loss_list.append(current_loss)
        if (j+1) % 10 == 0:
            print(j+1)
            print(current_loss)

    intensity_far_field = model(slm_phase)

    intensity_far_field = np.array(intensity_far_field[1])
    
    intensity_far_field_big = np.array(intensity_far_field[0])

    computed_phase = np.array(slm_phase[0,0,:,:], dtype=np.float64)
    
    return computed_phase, intensity_far_field, intensity_far_field_big