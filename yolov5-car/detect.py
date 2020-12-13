import argparse

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        start_point1 = (110, 10)
        end_point1 = (185, 163)
        color_rec = (255, 255, 255)

        start_point2 = (185, 10)
        end_point2 = (260, 163)

        start_point3 = (260, 10)
        end_point3 = (333, 163)

        start_point4 = (333, 10)
        end_point4 = (405, 163)

        start_point5 = (405, 10)
        end_point5 = (479, 163)

        start_point6 = (479, 10)
        end_point6 = (552, 163)

        start_point7 = (552, 10)
        end_point7 = (628, 163)

        start_point2_1 = (110, 183)
        end_point2_1 = (185, 335)

        start_point2_2 = (185, 183)
        end_point2_2 = (260, 335)

        start_point2_3 = (260, 183)
        end_point2_3 = (333, 335)

        start_point2_4 = (333, 183)
        end_point2_4 = (405, 335)

        start_point2_5 = (405, 183)
        end_point2_5 = (479, 335)

        start_point2_6 = (479, 183)
        end_point2_6 = (552, 335)

        start_point2_7 = (552, 183)
        end_point2_7 = (628, 335)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh

            car_nums = 0
            park_nums = []

            for k in range(14):
                park_nums.append(0)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        plot_one_box(xyxy, im0, color=colors[int(cls)], line_thickness=3)
                        #print('start_point X:' + str(int(xyxy[0])) + ' Y:'+ str(int(xyxy[1])) + ' end_point X:'+ str(int(xyxy[2])) + ' Y:'+str(int(xyxy[3])))
                        car_nums = car_nums + 1
                        #center point
                        x_center = (int(xyxy[2]) - int(xyxy[0]))/2 + int(xyxy[0])
                        y_center = (int(xyxy[3]) - int(xyxy[1]))/2 + int(xyxy[1])
                        #print(x_center, y_center)

                        if x_center > start_point1[0] and x_center < end_point1[0] and y_center > start_point1[1] and y_center < end_point1[1]:
                            park_nums[0] = 1
                        if x_center > start_point2[0] and x_center < end_point2[0] and y_center > start_point2[1] and y_center < end_point2[1]:
                            park_nums[1] = 1
                        if x_center > start_point3[0] and x_center < end_point3[0] and y_center > start_point3[1] and y_center < end_point3[1]:
                            park_nums[2] = 1
                        if x_center > start_point4[0] and x_center < end_point4[0] and y_center > start_point4[1] and y_center < end_point4[1]:
                            park_nums[3] = 1
                        if x_center > start_point5[0] and x_center < end_point5[0] and y_center > start_point5[1] and y_center < end_point5[1]:
                            park_nums[4] = 1
                        if x_center > start_point6[0] and x_center < end_point6[0] and y_center > start_point6[1] and y_center < end_point6[1]:
                            park_nums[5] = 1
                        if x_center > start_point7[0] and x_center < end_point7[0] and y_center > start_point7[1] and y_center < end_point7[1]:
                            park_nums[6] = 1
                        if x_center > start_point2_1[0] and x_center < end_point2_1[0] and y_center > start_point2_1[1] and y_center < end_point2_1[1]:
                            park_nums[7] = 1
                        if x_center > start_point2_2[0] and x_center < end_point2_2[0] and y_center > start_point2_2[1] and y_center < end_point2_2[1]:
                            park_nums[8] = 1
                        if x_center > start_point2_3[0] and x_center < end_point2_3[0] and y_center > start_point2_3[1] and y_center < end_point2_3[1]:
                            park_nums[9] = 1
                        if x_center > start_point2_4[0] and x_center < end_point2_4[0] and y_center > start_point2_4[1] and y_center < end_point2_4[1]:
                            park_nums[10] = 1
                        if x_center > start_point2_5[0] and x_center < end_point2_5[0] and y_center > start_point2_5[1] and y_center < end_point2_5[1]:
                            park_nums[11] = 1
                        if x_center > start_point2_6[0] and x_center < end_point2_6[0] and y_center > start_point2_6[1] and y_center < end_point2_6[1]:
                            park_nums[12] = 1
                        if x_center > start_point2_7[0] and x_center < end_point2_7[0] and y_center > start_point2_7[1] and y_center < end_point2_7[1]:
                            park_nums[13] = 1

            #for k in range(14):
             #   print(park_nums[k], end=" ")

            print(park_nums)
            print("Cars: " + str(car_nums), end=" ")
            print("Spaces: " + str(14-car_nums))

            with open('cas_infor.txt', 'w', encoding='utf-8') as f:
                text = str(park_nums[0])+str(park_nums[1])+str(park_nums[2])+str(park_nums[3])+str(park_nums[4])+str(park_nums[5])+str(park_nums[6])+str(park_nums[7])+str(park_nums[8])+str(park_nums[9])+str(park_nums[10])+str(park_nums[11])+str(park_nums[12])+str(park_nums[13])+str(car_nums)
                f.write(text)

            f.close()
            # Print time (inference + NMS)
            #print('%sDone. (%.3fs)' % (s, t2 - t1))
           # car_num = 0
           # c_string = s.split()
           # if len(c_string) > 2 is not None:
            #    car_num = int(c_string[2])
            #if det is not None:
            #    print(car_num)

            #if car_num > 0:
            #    print(det)
            #print(im0.shape[:2])
            '''
            cv2.rectangle(im0, start_point1, end_point1, color_rec, 2)
            cv2.rectangle(im0, start_point2, end_point2, color_rec, 2)
            cv2.rectangle(im0, start_point3, end_point3, color_rec, 2)
            cv2.rectangle(im0, start_point4, end_point4, color_rec, 2)
            cv2.rectangle(im0, start_point5, end_point5, color_rec, 2)
            cv2.rectangle(im0, start_point6, end_point6, color_rec, 2)
            cv2.rectangle(im0, start_point7, end_point7, color_rec, 2)

            cv2.rectangle(im0, start_point2_1, end_point2_1, color_rec, 2)
            cv2.rectangle(im0, start_point2_2, end_point2_2, color_rec, 2)
            cv2.rectangle(im0, start_point2_3, end_point2_3, color_rec, 2)
            cv2.rectangle(im0, start_point2_4, end_point2_4, color_rec, 2)
            cv2.rectangle(im0, start_point2_5, end_point2_5, color_rec, 2)
            cv2.rectangle(im0, start_point2_6, end_point2_6, color_rec, 2)
            cv2.rectangle(im0, start_point2_7, end_point2_7, color_rec, 2)
            '''
            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        detect()

        #print(car_nums)
        #for k in range(14):
        #    print(park_nums[k], end=" ")
        #send_text()

        # Update all models
        # for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
        #    detect()
        #    create_pretrained(opt.weights, opt.weights)
