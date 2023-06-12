import numpy as np
import cv2 as cv
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torch
from gtts import gTTS
import os


def get_pred(frame):
    frame=np.moveaxis(frame,2,0)
    img=torch.from_numpy(frame)
    batch = [preprocess(img)]
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="red",
                            width=4, font_size=30)
    box=box.numpy()
    box=np.moveaxis(box,0,2)
    return box,labels

def say(objs):
    objs=list(set(objs))
    objs=". ".join(objs)
    mytext = f'Detected {objs}'
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("detected.mp3")
    os.system("mpg321 detected.mp3")

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()

preprocess = weights.transforms()

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH,100)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,50)
print(f'Camera Input Dim - Width: {cap.get(cv.CAP_PROP_FRAME_WIDTH)} Height: {cap.get(cv.CAP_PROP_FRAME_HEIGHT)}')
if not cap.isOpened():
    print("Cannot open camera")
    exit()
counter=0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray,labels=get_pred(frame)
    if counter%10==0:
        say(labels)

    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
    counter+=1
    print(counter)

cap.release()
cv.destroyAllWindows()


