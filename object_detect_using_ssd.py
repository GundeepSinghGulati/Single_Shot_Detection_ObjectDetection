
#importing libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio
import warnings
warnings.filterwarnings("ignore")

#define function which will do the detections
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    #four transformation to network
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2,0,1)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    #detections contains four elements['batch', 'number of classes-like dog, human, plane','number of occurence of class',[score,x0,y0,x1,y1]] 
    #if score is < 0.6 then it is not found else found x0,y0 are upper corner coordinates and x1,y1 are lower corner coordinates
    for i in range(detections.size(1)):
        j = 0 #occurence of class
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame
#create the SSD neural network
net  = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location=lambda storage, loc: storage))

#creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

#doing some object detection on video
reader = imageio.get_reader('People.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('p.mp4', fps = fps)
for i,frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)
writer.close()
   
            
            