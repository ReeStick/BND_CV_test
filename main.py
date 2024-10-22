import cv2
import argparse
from ultralytics import YOLO


class VideoProcessor:
    """
    A class to handle video processing, including reading video frames, 
    performing object detection using YOLOv8, and saving processed frames to an output video file.

    Methods:
        get_video_config(cap): Retrieves the FPS, width, and height from the video capture object.
        process_frame(frame, model): Processes a single frame to detect objects and draw bounding boxes.
        process_video(model, output_video='output.mp4'): Processes an entire video, applying detection to each frame.
    """

    def __init__(self, cap):
        """
        Initializes the VideoProcessor with a given video capture object.

        Args:
            cap (cv2.VideoCapture): The video capture object used to read video frames and retrieve video properties.
        """
        self.cap = cap
        self.fps, self.width, self.height = self.get_video_config()


    def get_video_config(self):
        """
        Retrieves the configuration of the video, including frames per second (FPS), width, and height.

        Returns:
            tuple: A tuple containing three elements:
                - fps (int): The frame rate (frames per second) of the video.
                - width (int): The width of the video frames.
                - height (int): The height of the video frames.
        """
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return fps, width, height


    def process_frame(self, frame, model):
        """
        Processes a single video frame by resizing it, performing object detection using a YOLOv8 model,
        and drawing bounding boxes around detected persons. The frame is resized to fit the model's
        input size and the bounding boxes are rescaled to the original frame dimensions.

        Args:
            frame (numpy.ndarray): The input video frame to be processed, represented as a NumPy array.
            model (YOLO): The YOLOv8 model used to run inference and detect objects in the frame.

        Returns:
            numpy.ndarray: The processed frame with bounding boxes and labels drawn on detected persons.
        """
        
        # Resize frame to 640x640 for YOLOv8 model
        frame_resized = cv2.resize(frame, (640, 640))
        results = model.predict(source=frame_resized, save=False, conf=0.5)

        # Loop through detection results and draw bounding boxes
        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf.item()
                class_id = int(box.cls.item())

                if class_id == 0: # Class ID for human
                    x1 = int(x1 * (self.width / 640))
                    y1 = int(y1 * (self.height / 640))
                    x2 = int(x2 * (self.width / 640))
                    y2 = int(y2 * (self.height / 640))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'Person: {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return frame


    def process_video(self, model, output_video='output.mp4'):
        """
        Processes an input video frame by frame, applies object detection using a YOLOv8 model,
        and saves the processed frames with bounding boxes to an output video file.

        Args:
            model (YOLO): The YOLOv8 model used to perform object detection on each frame of the video.
            output_video (str, optional): The path to the output video file where processed frames will be saved.
                                          Default is 'output.mp4'.

        Returns:
            None
        """
        
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = self.process_frame(frame, model)
            out.write(frame)

        self.cap.release()
        out.release()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="YOLOv8 Video Processing")
    parser.add_argument('--input', type=str, required=True, help="Path to the input video file")
    parser.add_argument('--model', type=str, default="yolov8n.pt", help="YOLOv8 model file path (default: yolov8n.pt)")
    parser.add_argument('--output', type=str, default="output.mp4", help="Path to save the output video (default: output.mp4)")
    args = parser.parse_args()

    # Load video and model
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {args.input}")
        return

    # Load the YOLOv8 model
    model = YOLO(args.model)  

    # Initialize video processor and process the video
    processor = VideoProcessor(cap)
    processor.process_video(model, args.output)

    print(f"Processing complete. Video saved to {args.output}")


if __name__ == "__main__":
    main()
