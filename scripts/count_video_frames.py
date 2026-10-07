import cv2

for name in ["crosswalk", "fourway", "night"]:
    path = f"dataset/{name}.avi"

    cap = cv2.VideoCapture(path)
    count = 0

    while True:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1

    cap.release()

    print(f"{name}: decoded_frames={count}")