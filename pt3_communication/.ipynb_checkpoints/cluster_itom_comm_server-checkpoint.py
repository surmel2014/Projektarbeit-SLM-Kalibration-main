
"""The Python implementation of the gRPC server to communicate with itom."""

from concurrent import futures
import logging
import numpy as np
import tensorflow as tf
import time

# gRPC Stuff (should work in TF 2.15)
import grpc
import cluster_itom_comm_pb2
import cluster_itom_comm_pb2_grpc

# custom packages
import citl_utils

from Layer_combined_v6_2_for_CITL import build_model

def nparray2Array_message(np_array: np.ndarray):
    datatype = str(np_array.dtype)
    try:
        height, width = np_array.shape
    except ValueError:
        height = np.squeeze(np_array.shape)
        width = 0
    return cluster_itom_comm_pb2.Array_message(datatype = datatype,
                                                   width = width,
                                                   height = height,
                                                   bytesarray=np_array.tobytes())
def Array_message2nparray(Array):
    datatype = Array.datatype
    height = Array.height
    width = Array.width
    data = np.frombuffer( Array.bytesarray, dtype= datatype)
    try:
        return np.reshape(data, (height, width))
    except ValueError:
        return np.reshape(data, (height))


def nparray2weight(np_array: np.ndarray, layer_number:int, layer_name:str):
    datatype = str(np_array.dtype)
    try:
        height, width = np_array.shape
    except ValueError:
        height = np.squeeze(np_array.shape)
        width = 0
    return cluster_itom_comm_pb2.Weight_layer(datatype = datatype,
                                                   layernumber = layer_number,
                                                   layername = layer_name,
                                                   height = height,
                                                   width = width,
                                                   bytesarray=np_array.tobytes())
def weight2nparray(weight):
    datatype = weight.datatype
    layer_number = weight.layernumber
    layer_name = weight.layername
    height = weight.height
    width = weight.width
    data = np.frombuffer( weight.bytesarray, dtype= datatype)
    try:
        return np.reshape(data, (height, width)), layer_number, layer_name
    except ValueError:
        return np.reshape(data, (height)), layer_number, layer_name

class Communication_Server(cluster_itom_comm_pb2_grpc.Communication_ServerServicer):
    """Provides methods that implement functionality of route guide server."""
    
    def __init__(self, port_id = "0.0.0.0:3456"):
        self.initialize_server(port_id)
        
        #initialize members
        self.target_intensity = np.empty(0)
        self.initial_phase = np.empty(0)
        self.computed_phase = np.empty(0)
        self.prediction = np.empty(0)
        self.prediction_big = np.empty(0)
        
        self.loss_list = []
        
    def initialize_server(self, port_id):
        gigabyte = 1024 ** 3 *2 -1 #max size
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', gigabyte),
            ('grpc.max_receive_message_length', gigabyte)
         ])
        cluster_itom_comm_pb2_grpc.add_Communication_ServerServicer_to_server(
            self, self.server)

        self.server.add_insecure_port(port_id)
        self.server.start()
        self.server.wait_for_termination()
        

    def _sendImage_intern(self,request, context):
        self.target_intensity = Array_message2nparray(request)
        print('Target Intensity send')
        return cluster_itom_comm_pb2.Empty()
    
    def _sendPhase_intern(self, request, context):
        self.initial_phase = Array_message2nparray(request)
        print('Initial Phase send')
        return cluster_itom_comm_pb2.Empty()
    
    def _getSLMPhase_intern(self, request, context):
        return nparray2Array_message(self.computed_phase)
    
    def _getPrediction_intern(self, request, context):
        return nparray2Array_message(self.prediction)
    
    def _getPrediction_big_intern(self, request, context):
        return nparray2Array_message(self.prediction_big)
    
    def _startSGD1_intern(self, request, context):
        size_slm_x = 1920
        size_slm_y = 1080
        size_image = 4320
        size_slm_padding_x = 3360 #3780,3360
        size_slm_padding_y = 3780
        self.loss_list = []
        self.computed_phase, self.prediction = citl_utils.SGD_initial_phase(self.target_intensity, size_slm_y, size_slm_x, size_image, size_slm_padding_y, size_slm_padding_x, self.loss_list)
        self.prediction_big = self.prediction
        return cluster_itom_comm_pb2.Empty()
    
    def _CreateModel_intern(self, request, context):
        self.model = build_model()
            
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=50,
            decay_rate=0.1)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=['mse', None], metrics=['mse'])

        self.model.summary()
        return cluster_itom_comm_pb2.Empty()
    
    def _startTraining_intern(self, request, context):
        image_intensity = self.target_intensity
        slm_phase = self.initial_phase

        slm_phase = np.expand_dims(np.expand_dims(slm_phase,0),0)
        #slm_phase = np.flip(slm_phase, 3)
        image_crop = np.expand_dims(np.expand_dims(image_intensity,0),0)
        far_field_empty = np.zeros((1, 1, 8640, 8640))
        self.model.fit(slm_phase, [image_crop, far_field_empty], batch_size=1, epochs=50, verbose=1)

        prediction_ls = self.model.predict(slm_phase)
        self.prediction = np.asarray(prediction_ls[1][0,0,:,:], dtype = np.float64)
        self.prediction_big = np.asarray(prediction_ls[0][0,0,:,:], dtype = np.float64)
        return cluster_itom_comm_pb2.Empty()
        
    def _startSGD_intern(self, request, context):
        self.loss_list = []
        self.computed_phase, self.prediction, self.prediction_big = citl_utils.SGD_with_model(self.model, 
                                              initial_phase = self.initial_phase,
                                              image_intensity = self.target_intensity,
                                              size_image = 1920,
                                              loss_list = self.loss_list)
        return cluster_itom_comm_pb2.Empty()
    
    def _getModelWeights_intern(self, request, context):
        
        layer_numbers = len(self.model.layers)
        
        weights_message = cluster_itom_comm_pb2.Weights_model()
        for i in range(layer_numbers):
            current_weights = self.model.layers[i].get_weights()
            layer_name = self.model.layers[i].name
            #print(layer_name)
            
            if not current_weights:
                pass #no weights in layer
            else:
                current_weights = np.asarray(current_weights)
                current_weight_as_message = nparray2weight(current_weights, i, layer_name)
                weights_message.weight.append(current_weight_as_message)

        return weights_message
    
    def _setModelWeights_intern(self, request, context):
        
        num_weights = len(request.weight)
        for i in range(num_weights):
            current_weights, layer_number, layer_name = weight2nparray(request.weight[i])
            self.model.layers[layer_number].set_weights(current_weights)
        
        return cluster_itom_comm_pb2.Empty()
    
    def _getLoss_intern(self, request, context):
        loss_np = np.asarray(self.loss_list)
        return nparray2Array_message(loss_np)
    
    def _makePrediction_intern(self, request, context):
        phase = Array_message2nparray(request)
        phase = np.expand_dims(np.expand_dims(phase,0),0)
        prediction_big = self.model(phase)[1][0,0,:,:].numpy()
        return nparray2Array_message(prediction_big)
    
    def _sendMultipleCameraImages(self, request_iterator, context):
        img_list = []
        for message in request_iterator:
            array = Array_message2nparray(message)
            array = np.expand_dims(array,0)
            img_list.append(array)
            
        self.cameraImages = np.asarray(img_list)
        return cluster_itom_comm_pb2.Empty()
    
    def _sendMultipleSLMPhases(self, request_iterator, context):
        phase_list = []
        for message in request_iterator:
            array = Array_message2nparray(message)
            array = np.expand_dims(array,0)
            phase_list.append(array)
            
        self.slmPhases = np.asarray(phase_list)
        return cluster_itom_comm_pb2.Empty()
    
    def _fitAberrations(self, request, context):
        num_images = self.slmPhases.shape[0]
        dummy_far_field = np.zeros((num_images,1,8640,8640)) #TODO 8640 aus anderen Zahlen errechnen
        self.model.fit(self.slmPhases, [self.cameraImages, dummy_far_field], batch_size=2, epochs=200)
        
        return cluster_itom_comm_pb2.Empty()



if __name__ == "__main__":
    logging.basicConfig()
    serve()
