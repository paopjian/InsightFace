import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection','recognition'], providers=['CUDAExecutionProvider']) # CPUExecutionProvider CUDAExecutionProvider
# app.prepare(ctx_id=0, det_size=(640, 640))
app.prepare(ctx_id=0, det_thresh=0.25)

image1_path = "../data/lfw_funneled/Abel_Pacheco/Abel_Pacheco_0004.jpg"

img = cv2.imread(image1_path)
faces = app.get(img)

max_area = 0
max_face = None
for face in faces:
    keypoints = face["kps"]
    bbox = face["bbox"]
    all_non_negative = all(x >= 0 and y >= 0 for x, y in keypoints)
    if all_non_negative:
        area = np.abs(bbox[2]-bbox[0]) * np.abs(bbox[3]-bbox[1])
        if area > max_area:
            max_face = face
            max_area = area
faces = [max_face]

rimg = app.draw_on(img, faces)
cv2.imwrite("./output.jpg", rimg)


