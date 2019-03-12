import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np


# label : [[x11, y11, x12, y12],[x21, y21, x22, y22], ..]
def img_aug(image, label):
    ia.seed(1)

    bb = []
    for i in range(len(label)):
        bb.append(ia.BoundingBox(x1=int(label[i][0]), y1=int(label[i][1]), x2=int(label[i][2]), y2=int(label[i][3])))

    bbs = ia.BoundingBoxesOnImage(
        bb
        , shape=image.shape)

    mul1 = np.random.uniform(0.2, 1.8)
    mul2 = np.random.uniform(0.2, 1.8)
    trans1 = np.random.uniform(-50, 50)
    trans2 = np.random.uniform(-50, 50)
    scale1 = np.random.uniform(0.5, 1)
    scale2 = np.random.uniform(0.5, 1)

    seq = iaa.Sequential([
        iaa.Multiply((mul1, mul2)),  # change brightness, doesn't affect BBs
        iaa.Affine(
            scale=(scale1, scale2),
            translate_px={"x": round(trans1), "y": round(trans2)}
        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])

    # Make our sequence deterministic.
    # We can now apply it to the image and then to the BBs and it will
    # lead to the same augmentations.
    # IMPORTANT: Call this once PER BATCH, otherwise you will always get the
    # exactly same augmentations for every batch!
    seq_det = seq.to_deterministic()

    # Augment BBs and images.
    # As we only have one image and list of BBs, we use
    # [image] and [bbs] to turn both into lists (batches) for the
    # functions and then [0] to reverse that. In a real experiment, your
    # variables would likely already be lists.
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    # print coordinates before/after augmentation (see below)
    # use .x1_int, .y_int, ... to get integer coordinates

    bboxs = []
    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        # print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
        #     i,
        #     before.x1, before.y1, before.x2, before.y2,
        #     after.x1, after.y1, after.x2, after.y2)
        #     )
        bboxs.append([after.x1, after.y1, after.x2, after.y2])

    # image with BBs before/after augmentation (shown below)
    # image_before = bbs.draw_on_image(image, thickness=2)
    # image_after = bbs_aug.draw_on_image(image_aug, thickness=2, color=[0, 0, 255])

    # f, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(image_before)
    # ax2.imshow(image_after)
    # plt.show()
    return image_aug, bboxs


if __name__ == '__main__':
    image = ia.quokka(size=(256, 256))
    img_aug(image, [[65, 100, 200, 150],[150, 80, 200, 130]])
