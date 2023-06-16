import numpy as np
import cv2
import albumentations as A

size=1024

def vis_keypoints(image, bboxes, keypoints, path):
    color=(0, 255, 0)
    diameter=15
    thickness=5

    image = image.copy()

    for bbox in np.array(bboxes).astype(int):
        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, color, -1)
    cv2.imwrite(path, image)

im = cv2.imread('data/images/stamp_image.jpg')
assert im is not None
labels = np.array([
    [          0,      423.25,      269.89,      670.72,      276.14,      669.58,      731.38,      428.37,      731.38],
    [          0,      666.74,      268.18,      901.69,       260.8,      895.43,      751.84,      668.44,      733.65]])

transform = A.Compose([
        # A.RandomSizedCrop(min_max_height=(500, 800), width=size, height=size, p=1),
        # A.RandomSizedBBoxSafeCrop(width=size, height=size, p=1)
        A.BBoxSafeRandomCrop(p=1),
        A.Resize(width=size, height=size, p=1),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=[]),
        keypoint_params=A.KeypointParams(format='xy')
        )

xs = labels[:, 1::2].flatten()
ys = labels[:, 2::2].flatten()
# Make one bounding box in Albumentation format from all the polygons.
bboxes = [(xs.min(), ys.min(), xs.max(), ys.max())]
keypoints = [(x[0], x[1]) for x in labels[:, 1:].reshape((-1, 2))]  # remove class.

vis_keypoints(im, bboxes, keypoints, 'data/images/stamp_image_output_before.jpg')

new = transform(image=im, bboxes=bboxes, keypoints=keypoints)

vis_keypoints(new['image'], new['bboxes'], new['keypoints'], 'data/images/stamp_image_output_after.jpg')




