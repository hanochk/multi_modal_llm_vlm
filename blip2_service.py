import requests
import sys
sys.path.insert(0, './')
from service_interface import ServiceInterface
from PIL import Image
from typing import List
import numpy as np
import json

class BLIP2Service(ServiceInterface):
    def __init__(self, server_address='http://184.105.3.17:8087/infer'):
        self.blip2_service_ip = server_address #'http://127.0.0.1:8087/infer'  'http://184.105.3.17:8087/infer'
        self.headers = {
            'accept': '*/*',
            # requests won't add a boundary if this header is set when you pass files=
            # 'Content-Type': 'multipart/form-data',
        }
    
    def get_url_response(self, image_url, question, bboxes=[]):
        if isinstance(question, str): # Handle Single Input
            question = [[question]]
        if isinstance(question, list): # Handle Multiple/Batched Input
            if isinstance(question[0], str):
                question = [question]
        question = json.dumps(question)
        bboxes = json.dumps(bboxes)
        files = {
            'image': (None, image_url),
            'question': (None, question),
            'bboxes': (None, bboxes)
        }
        #print(files)
        try:
            response = requests.post(self.blip2_service_ip, headers=self.headers, files=files)
            #print(response.json())
            return [response.json()['answer']][0]
        except Exception as e: 
            return None
    
    def get_image_response(self, image : Image, text : List[str]) -> List[str]:
        raise Exception("Use URLs only.")

def main():
    server_address='http://209.51.170.37:8087/infer'
    blip2_service = BLIP2Service(server_address)

    # # Inputs
    url = 'http://74.82.29.209:9000/datasets/media/services/blip2_upload/3036_IN_TIME_00.28.05.713-00.28.23.353__286.jpg'
    texts = ["Enumerate all visual attributes of a {} shown in the image?"]
    bboxes = [[53.04999923706055, 199.6999969482422, 127.80999755859375, 396.29998779296875], [100.77999877929688, 209.41000366210938, 170.8800048828125, 380.7200012207031], [632.1799926757812, 124.19999694824219, 1019.760009765625, 743.0499877929688], [1599.1800537109375, 372.8900146484375, 1624.800048828125, 401.55999755859375], [1780.93994140625, 332.8999938964844, 1864.1199951171875, 438.6600036621094], [1406.1400146484375, 331.239990234375, 1527.050048828125, 444.1300048828125], [1610.6300048828125, 349.67999267578125, 1672.9599609375, 440.4100036621094], [1542.489990234375, 348.260009765625, 1655.1400146484375, 442.3699951171875], [744.3300170898438, 520.1500244140625, 843.3699951171875, 748.6900024414062], [996.4600219726562, 220.19000244140625, 1397.4300537109375, 940.010009765625], [979.1500244140625, 620.7100219726562, 1524.0, 948.0], [18.440000534057617, 727.5999755859375, 956.8499755859375, 948.969970703125], [928.1699829101562, 759.6500244140625, 1581.5799560546875, 945.5599975585938]]
    outputs = blip2_service.get_url_response(url, texts, bboxes)
    print("\n" + "#"*10)
    print("URL: {}".format(url))
    print("Texts: {}".format(texts))
    print("BBoxes: {}".format(bboxes))
    print("BLIP2 Service Outputs: {}".format(outputs))

    # Inputs
    # url = 'http://74.82.29.209:9000/datasets/media/frames/actionclipautoautotrain00010/frame0081.jpg'
    # texts = ["Question: What kind of clothes is the person in the picture wearing? Answer:"]
    # bboxes = [[394.0, 0.0, 517.0, 159.0], [374.0, 0.0, 510.0, 159.0]]
    # outputs = blip2_service.get_url_response(url, texts, bboxes)[0]
    # print("\n" + "#"*10)
    # print("URL: {}".format(url))
    # print("Texts: {}".format(texts))
    # print("BBoxes: {}".format(bboxes))
    # print("BLIP2 Service Outputs: {}".format(outputs))

    # # Inputs
    # url = 'http://74.82.29.209:9000/datasets/media/frames/actionclipautoautotrain00010/frame0081.jpg'
    # texts = ["Question: What kind of clothes is the person in the picture wearing? Answer:"]
    # bboxes = [394.0, 0.0, 517.0, 159.0]

    # outputs = blip2_service.get_url_response(url, texts, bboxes)[0]
    # print("\n" + "#"*10)
    # print("URL: {}".format(url))
    # print("Texts: {}".format(texts))
    # print("BBoxes: {}".format(bboxes))
    # print("BLIP2 Service Outputs: {}".format(outputs))

    # # # Inputs
    # url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    # texts = ["Question: How many dogs are in the picture?"]
    # bboxes = []

    # outputs = blip2_service.get_url_response(url, texts, bboxes)[0]
    # print("\n" + "#"*10)
    # print("URL: {}".format(url))
    # print("Texts: {}".format(texts))
    # print("BBoxes: {}".format(bboxes))
    # print("BLIP2 Service Outputs: {}".format(outputs))

if __name__ == "__main__":
    main()
