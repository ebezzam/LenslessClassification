"""
pip install -U yolov5

just for faces: https://github.com/elyha7/yoloface
git clone
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:/home/bezzam/LenslessClassification/yoloface"
"""


import torch
import numpy as np
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


detect_face = True
verbose = False
min_face = 100


# get all files from a directory
path = [
    "data/celeba_recovered_scene2mask0.55_height0.27_384x512",
    "data/celeba_recovered_scene2mask0.55_height0.27_384x512_diff_slm",
    "data/celeba_recovered_scene2mask0.55_height0.27_192x256",
    "data/celeba_recovered_scene2mask0.55_height0.27_192x256_diff_slm",
    "data/celeba_recovered_scene2mask0.55_height0.27_96x128",
    "data/celeba_recovered_scene2mask0.55_height0.27_96x128_diff_slm",
    "data/celeba_recovered_scene2mask0.55_height0.27_48x64",
    "data/celeba_recovered_scene2mask0.55_height0.27_48x64_diff_slm",
    "data/celeba_recovered_scene2mask0.55_height0.27_24x32",
    "data/celeba_recovered_scene2mask0.55_height0.27_24x32_diff_slm"
]


if detect_face:

    import sys
    sys.path.append("/home/bezzam/LenslessClassification/yoloface")
    from face_detector import YoloDetector

    # simple face detection
    model = YoloDetector(min_face=min_face)

    # -- single image
    # img = 'data/celeba_recovered_scene2mask0.55_height0.27_384x512/0.png'
    # # img = 'data/celeba_recovered_scene2mask0.55_height0.27_384x512_diff_slm/1.png'
    # orgimg = np.array(Image.open(img))
    # orgimg = orgimg[:,:, np.newaxis]
    # orgimg = np.concatenate((orgimg, orgimg, orgimg), axis=2)
    # print(orgimg.shape)

    # bboxes,points = model.predict(orgimg)
    # # print(bboxes)
    # # print(points)

    # # draw bounding box on image
    # plt.imshow(orgimg[:, :, 0], cmap='gray')
    # for bbox in bboxes[0]:
    #     x1, y1, x2, y2 = bbox
    #     plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-')
    # plt.savefig('test.png')

    # -- loop over files
    for p in path:

        # extract scenario name
        scenario = os.path.basename(p)
        print("\n--------------------")
        print(scenario)

        # get all files from a directory
        imgs = glob.glob(os.path.join(p, "*.png"))

        # remove files that end with "_raw.png"
        imgs_list = [i for i in imgs if not i.endswith("_raw.png")]

        count = 0
        for img in imgs_list:
            orgimg = np.array(Image.open(img))
            orgimg = orgimg[:,:, np.newaxis]
            orgimg = np.concatenate((orgimg, orgimg, orgimg), axis=2)
            bboxes,points = model.predict(orgimg)

            if len(np.squeeze(bboxes)) > 0:
                count += 1

        print(count / len(imgs_list) * 100, "%")

else:

    import yolov5

    # # Model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model = yolov5.load('fcakyon/yolov5s-v7.0')
    # model = yolov5.load('yolov5s.pt')
    
    # set model parameters
    model.conf = 0.3  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 5  # maximum number of detections per image

    # # single image
    # img = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    # results = model(img)
    # results.print()
    # results.save()


    for p in path:

        # extract scenario name
        scenario = os.path.basename(p)
        print("\n--------------------")
        print(scenario)

        # get all files from a directory
        imgs = glob.glob(os.path.join(p, "*.png"))

        # remove files that end with "_raw.png"
        imgs = [i for i in imgs if not i.endswith("_raw.png")]

        # Inference
        results = model(imgs)

        # Results
        if verbose:
            results.print()
            results.save()

        confidence_vals = []
        detection_count = 0
        for i in range(len(results.xyxy)):

            # filter pandas dataframe by column value
            df = results.pandas().xyxy[i]
            df = df[df['name'] == "person"]

            if len(df) > 0:
                confidence_vals.append(df.iloc[0]["confidence"])

                # check that big enough
                face_size = max(df.iloc[0]["xmax"] - df.iloc[0]["xmin"], df.iloc[0]["ymax"] - df.iloc[0]["ymin"])
                if face_size > min_face:
                    detection_count += 1


        # print results
        print("Average confidence:", np.mean(confidence_vals))
        print("Detection percentage:", detection_count / len(imgs) * 100, "%")