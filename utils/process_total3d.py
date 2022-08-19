import argparse
import pickle
import os
from PIL import Image
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Script for Total 3D Input')
    # The locationi of training set
    parser.add_argument('--input_pkl', type=str, required=True,
                        help='.pkl file')

    args = parser.parse_args()

    NYU40CLASSES = ['void',
                'wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'blinds', 'desk', 'shelves',
                'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
                'person', 'night_stand', 'toilet', 'sink', 'lamp',
                'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']

    dataPath = args.input_pkl
    with open(dataPath, 'rb') as f:
        data = pickle.load(f)
    # data: rgb_img, depth_map, boxes, camera, layout, sequence_id
    rgb_img           = data['rgb_img']
    intrinsic_mat     = data['camera']['K']
    boxes = data['boxes']
    # print(boxes['size_cls'])
    # print(boxes['bdb2D_pos'])
    detection_list = []
    for idx, cls_id in enumerate(boxes['size_cls']):
        if cls_id != 0:
            bbox = boxes['bdb2D_pos'][idx]
            det = {"bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "class": NYU40CLASSES[cls_id]}
            detection_list.append(det)

    # output folder
    outputDir = args.input_pkl.replace('.pkl', '')
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # save rgb img
    im = Image.fromarray(rgb_img)
    im.save(os.path.join(outputDir, 'img.jpg'))

    # save camera K
    # print(intrinsic_mat)
    with open(os.path.join(outputDir, 'cam_K.txt'), 'w') as f:
        f.write('%.1f %.1f %.1f\n' % (intrinsic_mat[0, 0], intrinsic_mat[0, 1], intrinsic_mat[0, 2]))
        f.write('%.1f %.1f %.1f\n' % (intrinsic_mat[1, 0], intrinsic_mat[1, 1], intrinsic_mat[1, 2]))
        f.write('%.1f %.1f %.1f\n' % (intrinsic_mat[2, 0], intrinsic_mat[2, 1], intrinsic_mat[2, 2]))

    # save detection.json
    # print(detection_list)
    with open(os.path.join(outputDir, 'detections.json'), 'w') as f:
        json.dump(detection_list, f)
