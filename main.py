import cv2
from ultralytics import YOLO

model = YOLO('best.pt')

# Width Height
exp_size = (960, 540)

title = "Summer Pockets YOLO"
cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(title, exp_size[0], exp_size[1])

frames = model.predict(source=1, stream=True, imgsz=(exp_size[1], exp_size[0]))

for frame in frames:
    cv2.imshow(title, frame.plot())
    cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1)

cv2.destroyAllWindows()
