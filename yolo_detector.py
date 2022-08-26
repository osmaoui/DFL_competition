from yolov6.core.inferer import Inferer
import cv2
import logging


logger = logging.getLogger()
class OfflineObjectDetectionYolo:
    def __init__(self, model_path, config_path, device, skip_frames, logger=None):
        self.yolo_model = Inferer(model_path, device, config_path, img_size=[1920,1080], half=False)
        self._conf_thresh = 0.25
        self._iou_thresh = 0.45
        self._skip_frames = skip_frames

    def predict(self, img):
        x = self.yolo_model.infer(
            img,
            self._conf_thresh,
            self._iou_thresh,
            classes=None,
            agnostic_nms=False,
            max_det=1000,
            save_dir=None,
            save_txt=False,
            save_img=False,
            hide_labels=False,
            hide_conf=False,
        )
        return x

if __name__ == "__main__":
    yolo_detctor = OfflineObjectDetectionYolo(
        model_path='/media/oussama/60d0458f-2f1f-4c73-bfe4-93757a0b94c5/home/oussama/workspace/reeplayer/github/reeplayer-AI---Tools/yolo_training/weights/yolov6s.pt',
        config_path='/media/oussama/60d0458f-2f1f-4c73-bfe4-93757a0b94c5/home/oussama/workspace/reeplayer/github/reeplayer-AI---Action-Tracking---Python-GPU/conf/coco.yaml',
        device="cpu",
        skip_frames=1,
        logger=logger
    )
    detections = yolo_detctor.predict(input_video='dfl-bundesliga-data-shootout/test/0b1495d3_0.mp4')