# importing libraries
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    
    
    ## construct a blob with the dimensions of the frame
    (h, w) = frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (224,224), (104.0, 177.0, 123.0))
    
    
    ## pass the blob through the network to obtain the detection
    faceNet.setInput(blob)
    detections=faceNet.forward()
    print(detections.shape)

    ##initialize faces and locations and the list of predictions
    faces=[]
    locs=[]
    preds=[]

    ## loop oer the detections
    for i in range(0,detections.shape[2]):
        
        
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0,0,i,2]
        
        
        ## filter out weak detections by ensuring confidence greater than minimum
        if confidence>0.5:
            
            
            #compute the[x,y] coordinates of the bounding boxfor the object
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")
            
            
            ##ensuring bounding boxes fall within directions of the frame
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX),min(h-1,endY))
            
            
            ##extracting the face, converting it from BGR to RGB, ordering, resizing to 224x224, processing
            face=frame[startY:endY,startX:endX]
            face=cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face=cv2.resize(face, (224,224))
            face=img_to_array(face)
            face=preprocess_input(face)


            #add face and boounding boxes to respective lists
            faces.append(face)
            locs.append((startX,startY,endX,endY))

    ## only make a prediction if only one fae is detected
    if len(faces)>0:
        
        
        ##we make batch predictions on all faces at the same timerather than one-by-one predictions
        ##this is a faster approach
        faces=np.array(faces,dtype="float32")
        preds=maskNet.predict(faces, batch_size=32)

    ##turn a 2 tuple for face locations and their corresponding locations
    return(locs,preds)

##loading serialized face detector from disk
protoxPath=os.path.expanduser("~")+"/Maskk/face_detector/deploy.prototxt"
weightsPath=os.path.expanduser("~")+"/Maskk/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(protoxPath,weightsPath)


## load face mask detector model from disk
maskNet =load_model("mask_detector.model")

##initialize video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

##loop over frames from video stream
while True:
    ##grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
    frame=vs.read()
    frame=imutils.resize(frame,width=400)

    ##detect faces in the frame and determine if they ae wearing a mask or not
    (locs, preds)=detect_and_predict_mask(frame,faceNet,maskNet)

    ##loop over the detected face locations and their corresponding locations
    for(box,pred) in zip(locs,preds):
        (startX,startY,endX,endY)=box
        (Mask,withoutMask)=pred

        ##determine class label and color we'll use to draw the bounding box and text
        label="No Mask" if Mask>withoutMask else "Mask"
        color=(255,0,0) if label==Mask else (0,0,255)

        ##include probability
        cv2.putText(frame,label,(startX, startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
        cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
    
    #show output frame
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1) & 0xFF

    #quit by pressing q
    if key==ord('q'):
        break
    

##cleanup
cv2.destroyAllWindows()
vs.stop()
