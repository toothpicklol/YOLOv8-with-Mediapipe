import torch
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import mediapipe  as mp
from ultralytics import YOLO
import cv2


gif_file = 'walk.gif'
cap = cv2.VideoCapture(gif_file)

# Check if the file was opened successfully
if not cap.isOpened():
    print("Error: Could not open GIF file.")
    exit()


mp_drawing = mp.solutions.drawing_utils         # mediapipe drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles # mediapipe drawing_styles
mp_holistic = mp.solutions.holistic

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5

model = YOLO("yolov8n.pt")  # choose model
count=0

with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
    while True:

        # Read a frame from the GIF
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            break
        # im2 = cv2.imread("man.jpg")
        im2=frame
        final = im2
        results = model(source=im2, classes=0)  # classes =0 only check type = "person"  #save_crop=True, save_txt=True,

        box_arr = []  # yolo
        # loc_array=[]  # original loc with yolo
        mp_array = []  # mediapipe loc with cut by original img need change

        # get yolo results loc
        for i in results[0].boxes.xywhn:
            box_arr.append(f"{i[0].item()} {i[1].item()} {i[2].item()} {i[3].item()}")

        dh, dw, _ = im2.shape
        for i in box_arr:

            # transform to original img loc

            x_center, y_center, w, h = i.strip().split()
            x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
            x_center = round(x_center * dw)
            y_center = round(y_center * dh)

            w = round(w * dw)
            h = round(h * dh)
            x = round(x_center - w / 2)
            y = round(y_center - h / 2)
            #  crop img and append loc
            # loc_array.append([w,h,x,y])
            imgCrop = im2[y:y + h, x:x + w]

            #  drawing mediapipe
            frame = imgCrop
            img = frame
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 將 BGR 轉換成 RGB
            results = holistic.process(img2)  # 開始偵測全身
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())

            # Overwrite the original image
            final[y:y + h, x:x + w] = img
            # cv2.imshow("a", final)


            # get erery pose_landmarks and transform to original img loc
            landmark_list = results.pose_landmarks
            image_rows, image_cols, _ = frame.shape
            idx_to_coordinates = {}

            if landmark_list != None:
                for idx, landmark in enumerate(landmark_list.landmark):
                    if ((landmark.HasField('visibility') and
                         landmark.visibility < _VISIBILITY_THRESHOLD) or
                            (landmark.HasField('presence') and
                             landmark.presence < _PRESENCE_THRESHOLD)):
                        continue
                    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
                    if landmark_px:
                        idx_to_coordinates[idx] = landmark_px
                if len(idx_to_coordinates) != 0:
                    mp_array.append(idx_to_coordinates)

                    # transform to original loc
                    for j in idx_to_coordinates:
                        # original x
                        print(x + idx_to_coordinates.get(j)[0])
                        # original y
                        print(y + idx_to_coordinates.get(j)[0])
                        
                    # Because some points cannot be determined,
                    # the number of each point has a corresponding position
                    # cannot append to array.
        cv2.imwrite("./gif/" + str(count) + ".jpg", final)
        count += 1






# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()


