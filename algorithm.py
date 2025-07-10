import numpy
import cv2


class Point(object):
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    @property
    def pos(self):
        return self.x, self.y

    @staticmethod
    def distance(a, b):
        return (a.x - b.x) ** 2 + (a.y - b.y) ** 2

    @staticmethod
    def center(a, b):
        return Point((a.x + b.x) / 2, (a.y + b.y) / 2)


class Rectangle(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @property
    def center(self):
        return Point.center(self.a, self.b)

    def inside(self, t):
        if self.a.x < t.x & t.x < self.b.x & self.a.y < t.y & t.y < self.b.y:
            return True
        else:
            return False


class QuadraticCurve(object):
    """An class for QuadraticCurve Approching
    Currently for testing only
    """
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy

    @classmethod
    def fit(cls, array_x, array_y):
        tick = numpy.asarray(range(array_x.__len__()))
        coef_x = numpy.polyfit(tick, array_x, 2)
        coef_y = numpy.polyfit(tick, array_y, 2)
        return cls(coef_x, coef_y)

    def distance(self, target: Point):
        # 2A^2+2D^2
        p_rd = 2*numpy.pow(self.cx[0], 2) + 2*numpy.pow(self.cy[0], 2)
        # 3AB+3DE
        p_nd = 3*self.cx[0]*self.cx[1] + 3*self.cy[0]*self.cy[1]
        # 2DF+E^2+2AC+B^2
        p_st = 2*self.cy[0]*self.cy[2] + numpy.pow(self.cy[1], 2) + 2*self.cx[0]*self.cx[2] + numpy.pow(self.cx[1], 2)
        # -2Dy-2Ax
        p_st = p_st - 2*self.cy[0]*target.y - 2*self.cx[0]*target.x
        # B(C-x)+E(F-y)
        p_cc = self.cx[1]*(self.cx[2] - target.x) + self.cy[1]*(self.cy[2] - target.y)
        # root
        roots_t = numpy.roots([p_rd, p_nd, p_st, p_cc])
        # verify
        dist = [Point.distance(target, Point(self.x(tick), self.y(tick))) for tick in roots_t]
        return min(dist)

    def x(self, tick):
        return self.cx[0] * (numpy.pow(tick, 2)) + self.cx[1] * tick + self.cx[2]

    def xi(self, tick):
        return int(self.x(tick))

    def y(self, tick):
        return self.cy[0] * (numpy.pow(tick, 2)) + self.cy[1] * tick + self.cy[2]

    def yi(self, tick):
        return int(self.y(tick))

    def gen(self, tick):
        return self.xi(tick), self.yi(tick)


class KalmanFilter(object):
    # predict the route using Kalman filter
    """
    balls_route_predict = []
    if balls_route.__len__() > 6:
        # setup
        karmal_filter = KalmanFilter()
        lastone = [karmal_filter.predict(ball) for ball in balls_route][-1]
        # predict
        # you can change the step
        for step in range(5):
            pt_point = karmal_filter.predict(lastone)
            balls_route_predict.append(pt_point)
            cv2.circle(image, pt_point.pos, 5, (255, 0, 0), -1)
            lastone = pt_point
    """
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0]], numpy.float32)
        self.kf.transitionMatrix = numpy.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], numpy.float32)

    def predict(self, current: Point):
        self.kf.correct(numpy.array([[numpy.float32(current.x)], [numpy.float32(current.y)]]))
        pt = self.kf.predict().tolist()
        return Point(pt[0][0], pt[1][0])
