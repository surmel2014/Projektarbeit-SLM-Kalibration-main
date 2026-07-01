
"""The Python implementation of the gRPC server."""

from concurrent import futures
import logging
import numpy as np

# gRPC Stuff (should work in TF 2.15)
import grpc
import pt3_comm_pb2
import pt3_comm_pb2_grpc

# custom packages
import display_class
from PIL import Image


def Array_message2nparray(Array):
    datatype = Array.datatype
    height = Array.height
    width = Array.width
    data = np.frombuffer( Array.bytesarray, dtype= datatype)
    try:
        return np.reshape(data, (height, width,3))
    except ValueError:
        return np.reshape(data, (height))



class Communication_Server(pt3_comm_pb2_grpc.Communication_ServerServicer):
    """Provides methods that implement functionality of route guide server."""
    
    def __init__(self, port_id = "0.0.0.0:3456"):
        self.display = display_class.ext_display(screen_id=2)
        
        print('initialized')
        
        self.initialize_server(port_id)
        
        self.points_holo = np.asarray(Image.open('mosaic_points_5x8.png'))

        self.holo_dark = np.asarray(Image.open('mosaic_dark.png'))
        
        #initialize members
        
    def initialize_server(self, port_id):
        gigabyte = 1024 ** 3 *2 -1 #max size
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', gigabyte),
            ('grpc.max_receive_message_length', gigabyte)
         ])
        pt3_comm_pb2_grpc.add_Communication_ServerServicer_to_server(
            self, self.server)

        self.server.add_insecure_port(port_id)
        self.server.start()
        self.server.wait_for_termination()
        

    def _send_and_display_image(self,request, context):
        self.mosaic = Array_message2nparray(request)
        print('mosaic received')
        self.display.show(self.mosaic)
        return pt3_comm_pb2.Empty()
    
    def _display_points(self, request, context):
        self.display.show(self.points_holo)
        return pt3_comm_pb2.Empty()

    def _display_dark(self, request, context):
        self.display.show(self.holo_dark)
        return pt3_comm_pb2.Empty()
    

if __name__ == "__main__":
    logging.basicConfig()
    
    cluster_instance = Communication_Server(port_id = "0.0.0.0:3456")#3556")
