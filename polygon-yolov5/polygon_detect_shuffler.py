import argparse
import shutil
import numpy as np
import progressbar
import ast

import torch
import torchvision

from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, set_logging, \
    polygon_non_max_suppression, polygon_scale_coords
from utils.torch_utils import select_device
import subprocess as sp

from shuffler.interface.pytorch.datasets import ImageDataset

def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    COMMAND1 = "nvidia-smi --query-gpu=memory.free --format=csv"
    COMMAND2 = "nvidia-smi --query-gpu=memory.used --format=csv"
    COMMAND3 = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND1.split()))[1:]
    memory_free_values = [int(x.split()[0])/1024 for i, x in enumerate(memory_free_info)]
    memory_used_info = _output_to_list(sp.check_output(COMMAND2.split()))[1:]
    memory_used_values = [int(x.split()[0])/1024 for i, x in enumerate(memory_used_info)]
    memory_total_info = _output_to_list(sp.check_output(COMMAND3.split()))[1:]
    memory_total_values = [int(x.split()[0])/1024 for i, x in enumerate(memory_total_info)]
    print(f'{"Used":18s}\t{"Free":18s}\t{"Total":18s}')
    for free, used, total in zip(memory_free_values, memory_used_values, memory_total_values):
        print(f'{used:.3f} {"GB": <6s}{used/total:<9.2%}\t{free:.3f} {"GB": <6s}{free/total:<9.2%}\t{total:.2f}')
        

@torch.no_grad()
def detect(weights='polygon-yolov5s-ucas.pt',  # model.pt path(s)
           in_db_file=None,
           rootdir=None,
           imgsz=640,  # inference size (pixels)
           conf_thres=0.25,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=1000,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           out_db_file=None,
           half=False,  # use FP16 half-precision inference
           coco_category_id_to_name_map={},
           ):
    progressbar.streams.wrap_stderr()
    progressbar.streams.wrap_stdout()
    FORMAT = '[%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s'

    labels_to_names = ast.literal_eval(coco_category_id_to_name_map)

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    # assert stride == 1
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    get_gpu_memory()

    # Polygon does not support second-stage classifier
    classify = False
    assert not classify, "polygon does not support second-stage classifier"

    transform_image = torchvision.transforms.Compose([
        lambda img: img.transpose((2, 0, 1)),
        lambda img: np.ascontiguousarray(img),
        lambda img: torch.Tensor(img),
        torchvision.transforms.Resize(imgsz),
    ])

    shutil.copyfile(in_db_file, out_db_file)
    dataset = ImageDataset(
        db_file=out_db_file,
        rootdir=rootdir,
        mode='w',
        used_keys=['image', 'imagefile', 'image_width', 'image_height'],
        transform_group={'image': transform_image})
    cursor = dataset.conn.cursor()

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    for sample in progressbar.progressbar(dataset):
        img = sample['image']
        imagefile = sample['imagefile']
        orig_shape = (sample['image_height'], sample['image_width'], 3)

        img = img.to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=augment)[0]

        # Apply polygon NMS
        pred = polygon_non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :8] = polygon_scale_coords(img.shape[2:], det[:, :8], orig_shape)

                # Write results
                for *xyxyxyxy, conf, cls in reversed(det):
                    object_entry = (imagefile, None, None, None, None, 
                                    labels_to_names[int(cls)], conf.item())
                    s = 'objects(imagefile,x1,y1,width,height,name,score)'
                    cursor.execute('INSERT INTO %s VALUES (?,?,?,?,?,?,?)' % s,
                                    object_entry)
                    objectid = cursor.lastrowid

                    xyxyxyxy = torch.tensor(xyxyxyxy).cpu().numpy().astype(float)
                    s = 'polygons(objectid,x,y)'
                    cursor.execute('INSERT INTO %s VALUES (?,?,?)' % s, (objectid, xyxyxyxy[0], xyxyxyxy[1]))
                    cursor.execute('INSERT INTO %s VALUES (?,?,?)' % s, (objectid, xyxyxyxy[2], xyxyxyxy[3]))
                    cursor.execute('INSERT INTO %s VALUES (?,?,?)' % s, (objectid, xyxyxyxy[4], xyxyxyxy[5]))
                    cursor.execute('INSERT INTO %s VALUES (?,?,?)' % s, (objectid, xyxyxyxy[6], xyxyxyxy[7]))

    dataset.conn.commit()
    dataset.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument(
        '-i',
        '--in_db_file',
        help='Path to Shuffler database with images to run inference on. '
        'Will write to the same db.',
        required=True)
    parser.add_argument(
        '-o',
        '--out_db_file',
        help=
        'The path to a new Shuffler database, where detections will be stored.',
        default='examples/detected/epoch10-test.db')
    parser.add_argument('--rootdir',
                        help='Where image files in the db are relative to.',
                        required=True)
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument(
        '--coco_category_id_to_name_map',
        default='{0: "page"}',
        help='A map (as a json string) from category id to its name.')

    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'thop'))

    detect(**vars(opt))
