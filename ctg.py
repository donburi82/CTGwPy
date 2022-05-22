try:
    from PIL import Image
except ImportError:
    import Image
from PIL import ImageDraw
import time
import easyocr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from cv2 import cv2
from colour import Color
from operator import itemgetter


class FhrVar:
    def __init__(self):
        self.level = 0
        self.image = np.zeros((209, 120, 3), dtype=np.uint8)

        self.red = Color("#0000ff")
        self.green = Color("#00ff00")
        self.color = list(self.red.range_to(self.green, 209))
        self.color_to_rgb()

    def change_fhr_level(self, level):
        self.level = level

        # change box
        self.image[:, :] = 0
        for index in range(209-self.level*1+30, 209):
            self.image[index, 45:75] = self.color[index]

        cv2.namedWindow('fhr_level')
        cv2.moveWindow('fhr_level', 650, 20)
        cv2.putText(self.image, str(self.level), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.imshow('fhr_level', self.image)

    def color_to_rgb(self):
        self.color = [tuple([int(z * 255) for z in x.rgb]) for x in self.color]


class TocoVar:
    def __init__(self):
        self.level = 0
        self.image = np.zeros((237, 120, 3), dtype=np.uint8)

        self.red = Color("#0000ff")
        self.green = Color("#00ff00")
        self.color = list(self.red.range_to(self.green, 237))
        self.color_to_rgb()

    def change_toco_level(self, level):
        self.level = level

        # change box
        self.image[:, :] = 0
        for index in range(237-self.level*3, 237):
            self.image[index, 45:75] = self.color[index]

        cv2.namedWindow('toco_level')
        cv2.moveWindow('toco_level', 650, 280)
        cv2.putText(self.image, str(self.level), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.imshow('toco_level', self.image)

    def color_to_rgb(self):
        self.color = [tuple([int(z * 255) for z in x.rgb]) for x in self.color]


def fhr_on_click(event, x, y, flags, param):
    global fhrPoints, click

    if event == cv2.EVENT_LBUTTONDOWN:
        fhrPoints = [(x, y)]
        click = True
    elif event == cv2.EVENT_LBUTTONUP:
        fhrPoints.append((x, y))
        click = False

        cv2.rectangle(frame, fhrPoints[0], fhrPoints[1], (0, 255, 0), 2)
        cv2.imshow("frame", frame)


def toco_on_click(event, x, y, flags, param):
    global tocoPoints, click

    if event == cv2.EVENT_LBUTTONDOWN:
        tocoPoints = [(x, y)]
        click = True
    elif event == cv2.EVENT_LBUTTONUP:
        tocoPoints.append((x, y))
        click = False

        cv2.rectangle(frame, tocoPoints[0], tocoPoints[1], (0, 255, 0), 2)
        cv2.imshow("frame", frame)


def draw_boxes(image, bounds):
    # 확률 높은 bound 하나만 선택
    bound = max(bounds, key=itemgetter(2))

    return image, bound


def move_figure(f, x, y):
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))


def RateLimited(maxPerSecond):
    minInterval = 1.0 / float(maxPerSecond)

    def decorate(func):
        lastTimeCalled = [0.0]

        def rateLimitedFunction(*args, **kwargs):
            elapsed = time.time() - lastTimeCalled[0]
            leftToWait = minInterval - elapsed
            if leftToWait > 0:
                time.sleep(leftToWait)
            ret = func(*args, **kwargs)
            lastTimeCalled[0] = time.time()
            return ret
        return rateLimitedFunction
    return decorate


def plot_values():
    try:
        counter.append(i)

        fhr_max = int(fhr_bound_max[1])
        fhrBox.change_fhr_level(fhr_max)
        fhr.append(fhr_max)
        axs[0].plot(counter, fhr, linewidth=1, color='k')
        #axs[0].plot(counter, fhr, color='y')

        toco_max = int(toco_bound_max[1])
        tocoBox.change_toco_level(toco_max)
        toco.append(toco_max)
        axs[1].plot(counter, toco, linewidth=1, color='k')
        #axs[1].plot(counter, toco, color='g')

        fig.canvas.draw()
    except ValueError as e:
        fhrBox.change_fhr_level(0)
        tocoBox.change_toco_level(0)


rate = input("Enter plot rate (in per second): ")
plot_values = (RateLimited(rate))(plot_values)

# Debugging
# DIR_READ = 'test.mp4'
# cap = cv2.VideoCapture(DIR_READ)

# External USB Camera
cap = cv2.VideoCapture(0)

reader = easyocr.Reader(['en'])
key = None
click = False

fhrPoints = [[]]
tocoPoints = [[]]

fhrROI = None
tocoROI = None

fhrBox = FhrVar()
tocoBox = TocoVar()

counter, fhr, toco = [], [], []
i = 0

fig, axs = plt.subplots(2, 1, constrained_layout=True)
# grid
axs[0].set_axisbelow(True)
axs[1].set_axisbelow(True)
axs[0].minorticks_on()
axs[1].minorticks_on()
axs[0].grid(which='major', linestyle='-', linewidth='0.4', color='black')
axs[1].grid(which='major', linestyle='-', linewidth='0.4', color='black')
axs[0].grid(which='minor', linestyle='-', linewidth='0.1', color='black')
axs[1].grid(which='minor', linestyle='-', linewidth='0.1', color='black')

# fhrAx == axs[0]
axs[0].set_title("FHR")
axs[0].set_ylim([30, 240])
axs[0].set_xlim(left=10, right=0)

# tocoAx == axs[1]
axs[1].set_title("Toco")
axs[1].set_ylim([0, 80])
axs[1].set_xlim(left=10, right=0)

move_figure(fig, 0, 0)
fig.suptitle("CTG", fontsize=16)
fig.show()

cv2.namedWindow('frame')
cv2.moveWindow('frame', 0, 0)

while cap.isOpened():

    ret, frame = cap.read()

    # fhr ROI
    while fhrROI is None:
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        # cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', fhr_on_click)

        if key == ord('c'):
            break

    fhrROI = frame[fhrPoints[0][1]:fhrPoints[1]
                   [1], fhrPoints[0][0]:fhrPoints[1][0]]
    fhrBounds = reader.readtext(fhrROI)
    fhr_result_image, fhr_bound_max = draw_boxes(fhrROI, fhrBounds)
    frame[fhrPoints[0][1]:fhrPoints[1][1],
          fhrPoints[0][0]:fhrPoints[1][0]] = fhrROI

    # toco ROI
    while tocoROI is None:
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        # cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', toco_on_click)

        if key == ord('c'):
            break

    # cv2.destroyWindow('frame')
    tocoROI = frame[tocoPoints[0][1]:tocoPoints[1]
                    [1], tocoPoints[0][0]:tocoPoints[1][0]]
    tocoBounds = reader.readtext(tocoROI)
    toco_result_image, toco_bound_max = draw_boxes(tocoROI, tocoBounds)
    frame[tocoPoints[0][1]:tocoPoints[1][1],
          tocoPoints[0][0]:tocoPoints[1][0]] = tocoROI

    # level change
    plot_values()

    # # show live stream of video
    # cv2.rectangle(frame, fhrPoints[0], fhrPoints[1],
    #               color=(0, 0, 255), thickness=2)
    # cv2.rectangle(frame, tocoPoints[0], tocoPoints[1],
    #               color=(0, 0, 255), thickness=2)
    # cv2.imshow('frame', frame)

    # axs[0].set_xlim(left=max(i, 10), right=max(0, i-10))
    # axs[1].set_xlim(left=max(i, 10), right=max(0, i-10))
    if i % 20 == 0:
        axs[0].set_xlim(left=i+20, right=i)
        axs[1].set_xlim(left=i+20, right=i)

    i += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
