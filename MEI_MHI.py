from skimage.measure import compare_ssim
import numpy as np
import imutils
import cv2
import pandas as pd

huarray=[]

def video():
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")
    
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('test.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    
    while(True):
        ret, frame = cap.read()
        
        if ret == True: 
            
            # Write the frame into the file 'output.avi'
            out.write(frame)
        
            # Display the resulting frame    
            cv2.imshow('frame',frame)
        
            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else:
            break 
        
    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()

def image_resize(path, width = None, height = None, inter = cv2.INTER_AREA):
    image=cv2.imread(path,0)
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def writefile(data,path,column):
    df=pd.DataFrame(np.array(data,dtype="object"),columns=column)
    with open(path,'w+') as f:
        df.to_csv(f,mode='w',header=False)

def calculatehumoments(image1,image2,label):
    lst=[]
    MEIarray=list(cv2.HuMoments(cv2.moments(image1)).flatten())
    MHIarray=list(cv2.HuMoments(cv2.moments(image2)).flatten())
    lst.append(label)
    for i in MEIarray:
        lst.append(i)
    huarray.append(lst)

def createMEIsandMHIs(path,i,j,k):
    cap=cv2.VideoCapture(path)
    firstFrame=None
    width,height=cap.get(3),cap.get(4)
    image1 = np.zeros((int(height), int(width)), np.uint8)
    image2 = np.zeros((int(height), int(width)), np.uint8)
    ctr=1
    while True:
        ret,frame=cap.read()
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if firstFrame is None:
            firstFrame = gray
            continue
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cv2.imshow('frame',thresh)
        image1=cv2.add(image1,thresh)
        image2=cv2.addWeighted(image2,1,thresh,ctr/1000,0)
        ctr+=1
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break
    cv2.imwrite("MEI%d%d%d.jpg"%(i,j,k),image1)
    cv2.imwrite("MHI%d%d%d.jpg"%(i,j,k),image2)
    image1=image_resize("MEI%d%d%d.jpg"%(i,j,k),height=120,width=90)
    cv2.imwrite("test.jpg",image1)
    calculatehumoments(image1,image2,i)
    cap.release()
    cv2.destroyAllWindows()
video()
createMEIsandMHIs('test.mp4',1,1,1)