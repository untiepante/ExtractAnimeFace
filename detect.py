import cv2
import sys
import os.path
import os
import glob

def detect(filename, cascade_file = "lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("AnimeFaceDetect", image)
    cv2.waitKey(0)
    cv2.imwrite("out.png", image)


def extract(filename, outpath, cascade_file = "lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    if (image is None):
        gif = cv2.VideoCapture(filename)
        _, image = gif.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))

    idx = 0
    for (x, y, w, h) in faces:
        cv2.imwrite(outpath+"/"+str(idx)+os.path.basename(filename) + ".png", image[y:y+h, x:x+w])
        idx+=1

    return len(faces)

# Extract cropped face, and **DELETE** the input images in which detector found any face.
dirpath = r"D:/crawled_imgs/*"
outdirpath = r"D:/crawled_imgs/extract"
imgPathes = glob.glob(dirpath)
idx = 0
for imgPath in imgPathes:
    if not (os.path.isfile(imgPath)):
        continue
    cntFaces = extract(imgPath, outdirpath)
    idx += 1
    if (cntFaces == 0):
        print("[{0}/{1}] '{2}': No face detected..?".format(idx, len(imgPathes), os.path.basename(imgPath)))
    else:
        # Delete the input file
        os.remove(imgPath)
        #print("[{0}/{1}] '{2}': {3} face extracted.".format(idx, len(imgPathes), os.path.basename(imgPath), cntFaces))
        #print("[{0}/{1}]".format(idx, len(imgPathes)))
        pass
    