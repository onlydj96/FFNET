import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import time
import requests
import argparse
from utils import pred_heatmap

def process_rtsp_stream(args):
    cap = cv2.VideoCapture(args.rtsp_url)
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream {args.rtsp_url}")
        return

    session = ort.InferenceSession(args.onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])


    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 15  # 2초당 1프레임 처리

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start = time.time()
        # 초당 1프레임 처리 (프레임 간격 맞추기)
        if frame_count % frame_interval != 0:
            continue

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        overlaying_time = time.time()
        pred_map = pred_heatmap(session, image)
        
        count = pred_map.sum()

        vis_img = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-5)
        vis_img = (vis_img * 255).astype(np.uint8)
        vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
        vis_img = cv2.resize(vis_img, (frame.shape[1], frame.shape[0]))

        overlay = cv2.addWeighted(frame, 0.6, vis_img, 0.4, 0)
        cv2.putText(overlay, f'Count: {int(count)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # # Display the resulting frame
        # cv2.imshow('Frame', overlay)

        # # Press Q on keyboard to exit the video early
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', overlay)

        if not ret:
            continue

        # Send to FastAPI server
        response = requests.post(args.server_url, files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")})
        print(response.status_code, response.reason) 

        print(f"Frame id : {frame_count} | Time : {round(time.time()-start, 3)}초")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtsp_url', default='rtsp://root:kiot!@34@192.168.20.101/axis-media/media.amp', type=str, help="RTSP URL of the stream to be processed.")
    parser.add_argument('--onnx_path', default='weights/density_estimation.onnx', type=str, help="the ONNX model path to be loaded")
    parser.add_argument('--server_url', default='http://0.0.0.0:5000/upload', type=str, help="Server Url that send the data to")
    
    args = parser.parse_args()
    process_rtsp_stream(args)
