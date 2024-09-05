import cv2
import argparse

from ultralytics import YOLO
import supervision as sv


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='FireDetector Live')
    parser.add_argument("--webcam-resolution", 
                        default=[640,480],
                          nargs=2,
                          type=int
    )
    args = parser.parse_args()
    return args

def main():

    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO('models/best.pt')

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()


        

        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.names[class_id]} {confidence:0.2f}"
            for _,confidence, class_id, _
            in detections

        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)


        cv2.imshow("FireDetection", frame)

        

        if(cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()


# things to implent: a threshold for the fire
# ex. take the confidence level, then actully report if if confidence > 0.3
#