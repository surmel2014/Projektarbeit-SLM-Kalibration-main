# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the gRPC route guide client."""

#https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal
#https://www.hardikp.com/2018/07/28/services/

from __future__ import print_function

import logging
import random

import numpy as np
import grpc
import pt3_communication.pt3_comm_pb2 as pt3_comm_pb2
import pt3_communication.pt3_comm_pb2_grpc as pt3_comm_pb2_grpc


def nparray2Array_message(np_array: np.ndarray):
    datatype = str(np_array.dtype)
    try:
        _, height, width, = np_array.shape
    except ValueError:
        height = np.squeeze(np_array.shape)
        width = 0
    
    return pt3_comm_pb2.Array_message(datatype = datatype,
                                                   width = width,
                                                   height = height,
                                                   bytesarray=np_array.tobytes())


class Communication_Class():

  def __init__(self, ip_port:str):
      gigabyte = 1024 ** 3* 2-1
      channel = grpc.insecure_channel(ip_port, options=[
                 ('grpc.max_send_message_length', gigabyte),
                 ('grpc.max_receive_message_length', gigabyte)])
      self.stub = pt3_comm_pb2_grpc.Communication_ServerStub(channel)
    

  def send_and_display_image(self, array):
    array4sending = nparray2Array_message(array)
    return self.stub._send_and_display_image(array4sending)

  def display_points(self):
    return self.stub._display_points(pt3_comm_pb2.Empty())

  def display_dark(self):
    return self.stub._display_dark(pt3_comm_pb2.Empty())


if __name__ == "__main__":
    logging.basicConfig()


    #cluster = Communication_Class(ip_port = "localhost:3456") #192.168.240.132:3456")#50051
    cluster = Communication_Class(ip_port = "192.168.240.131:3456") #192.168.240.132:3456")#50051
    from PIL import Image
    #holo = np.asarray(Image.open('./Motive/mosaic_bike.png'))
    #holo = np.asarray(Image.open('./Homography_/mosaic_points_5x8.png'))
    holo = np.asarray(Image.open('./Homography_/mosaic_dark.png'))
    #holo = np.asarray(Image.open('../Motive/mosaic_Cars-none_LcgsZero.png'))
    #holo = np.zeros((10,10))#np.asarray(Image.open('holo.png').convert('L'))/255
    #cluster.send_and_display_image(holo)
    print('send')
    cluster.display_dark()

    #cluster.sendPhase(np.zeros((1080,1920), dtype = np.float64))
    