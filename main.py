import cv2
import numpy
from ultralytics import YOLO
from algorithm import Point, Rectangle, QuadraticCurve
import pygetwindow as gw
import mouse

# Process Width Height
exp_size = (512, 288)
# Game Window Title
game_title = 'Summer Pockets REFLECTION BLUE'
# Chose your model
model = YOLO('best.pt')
# Chose your OBS Virtual Camera
cam = cv2.VideoCapture(1)


cv2.namedWindow("YOLO", cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("YOLO", exp_size[0], exp_size[1])
cam.set(cv2.CAP_PROP_FRAME_WIDTH, exp_size[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, exp_size[1])
# Find game
game_win = gw.getWindowsWithTitle(game_title)[0]
game_pos = (game_win.left, game_win.top)
game_size = (game_win.width, game_win.height)
xrt, yrt = game_size[0] / exp_size[0], game_size[1] / exp_size[1]
balls_route = []


def capture():
    _, frame = cam.read()
    frame = model.predict(source=frame, imgsz=(exp_size[1], exp_size[0]), device=0)[0]
    boxes = frame.boxes.data.cpu().numpy().tolist()
    # Find all balls and targets
    balls = filter(lambda xn: xn[5] == 0.0, boxes)
    balls = [Point.center(Point(box[0], box[1]), Point(box[2], box[3])) for box in balls]
    targets = filter(lambda xn: xn[5] == 1.0, boxes)
    targets = [Rectangle(Point(box[0], box[1]), Point(box[2], box[3])) for box in targets]
    return frame, balls, targets


if __name__ == '__main__':
    missing_tracker = 0
    while True:
        # Grab one frame
        frame, balls, targets = capture()
        image = frame.plot()
        # Clear route when missing too many balls
        balls_route.extend(balls)
        # Draw the route of balls
        [cv2.circle(image, ball.pos, 5, (0, 0, 255), -1) for ball in balls_route]
        # missing ball
        if missing_tracker >= 20:
            missing_tracker = 0
            balls_route.clear()
        if not balls:
            missing_tracker += 1
            # Display
            cv2.imshow("YOLO", image)
            cv2.setWindowProperty("YOLO", cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(1)
            continue
        # Approch the route of ball using QuadraticCurve
        if balls_route.__len__() > 10 and targets:
            arx = numpy.asarray([pt.x for pt in balls_route][-10:])
            ary = numpy.asarray([pt.y for pt in balls_route][-10:])
            equation = QuadraticCurve.fit(arx, ary)
            # [cv2.circle(image, equation.gen(tick), 2, (255, 0, 0), -1) for tick in range(-50, 50)]
            # calculate the most possilbe target
            t2b_dis = [equation.distance(t.center) for t in targets]
            t2b_maps = dict(zip(t2b_dis, targets))
            tg = t2b_maps[min(t2b_dis)]
            cv2.circle(image, tg.center.pos, 10, (255, 0, 0), -1)
            if Point.distance(tg.center, balls[-1]) < 2000:
                gm_x, gm_y = tg.center.x * xrt + game_pos[0], tg.center.y * yrt + game_pos[1]
                mouse.move(gm_x, gm_y)
                mouse.click()
                missing_tracker += 20
        # Display
        cv2.imshow("YOLO", image)
        cv2.setWindowProperty("YOLO", cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)
