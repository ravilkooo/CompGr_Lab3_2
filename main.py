import math
import random
import pygame
import os
import numpy as np

pygame.init()
W = 800
H = 600

sc = pygame.display.set_mode((W, H))
pygame.display.set_caption("CompGr_Lab3_2")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

FPS = 60
clock = pygame.time.Clock()


def rand_bezier():
    points = np.zeros((5, 2))
    for i in range(points.shape[0]):
        points[i][0] = random.randint(20, W - 20)
        points[i][1] = random.randint(20, H - 20)
    draw_bezier(points, 5)


def get_face_points(filename):
    points = np.zeros((0, 2))
    with open(filename, 'r') as file:
        line = file.readline()
        while line:
            x, y = line.split()
            points = np.append(points, [[int(x), int(y)]], axis=0)
            line = file.readline()
    return points


def draw_iter_bezier(points, d, deg, closed=False):
    new_size = points.shape[0] + (points.shape[0] // (deg - 2)) + (points.shape[0] % (deg - 2) != 0)
    extra = (deg - new_size % (deg - 1)) % (deg - 1)
    ext_points = np.zeros((new_size + extra, 2), dtype=float)
    ext_points[0] = (points[0] + points[-1]) / 2
    ext_points[1:deg] = points[0:deg - 1]
    for i in range(1, new_size // (deg - 1)):
        ext_points[i * (deg - 1)] = (points[i * (deg - 2) - 1] + points[i * (deg - 2)]) / 2
        ext_points[i * (deg - 1) + 1:(i + 1) * (deg - 1)] = points[i * (deg - 2):(i + 1) * (deg - 2)]
    to_pop = 0
    if new_size % (deg - 1) == 0:
        to_pop = 1
        ext_points[new_size] = (points[0] + points[-1]) / 2
    else:
        ext_points[(new_size // (deg - 1)) * (deg - 1)] = (points[(points.shape[0] // (deg - 2)) * (deg - 2) - 1] +
                                                           points[(points.shape[0] // (deg - 2)) * (deg - 2)]) / 2
        ext_points[(new_size // (deg - 1)) * (deg - 1) + 1:] = np.append(
            points[(points.shape[0] // (deg - 2)) * (deg - 2):],
            np.ones((extra, 1)) @ np.array([(points[0] + points[-1]) / 2]), axis=0)
        to_pop = extra
    parts = ext_points.shape[0] // (deg - 1)
    if not closed:
        ext_points = ext_points[0:-to_pop]
        ext_points[0] = ext_points[1]
        to_add = (1-ext_points.shape[0] % (deg - 1))
        ext_points = np.append(ext_points, np.ones((to_add, 1)) @ np.array([ext_points[-1]]), axis=0)
    for i in range(parts):
        draw_bezier(ext_points[(deg - 1) * i:(deg - 1) * (i + 1) + 1], d)


def draw_bezier(points, d):
    deg = points.shape[0] - 1
    if deg < 0: return
    l = np.sqrt(np.sum((points[0] - points[1]) * (points[0] - points[1])))
    for i in range(1, deg):
        l += np.sqrt(np.sum((points[i] - points[i + 1]) * (points[i] - points[i + 1])))
    n = int(l / d) if (l/d) > 10 else 10
    t = np.arange(n + 1) / n
    steps = list(bezier_func(points, deg, t))

    draw_lines_custom(steps)
    # draw_points_custom(points)
    # pygame.draw.lines(sc, (255, 0, 0), False, steps, 1)


def bezier_func(points, deg, t):
    C_n_k = lambda n, k: 1 if ((n < k) or (k < 0)) else (
            math.factorial(n) / (math.factorial(n - k) * math.factorial(k)))
    for t_i in t:
        res = np.array([0., 0.])
        for i in range(deg + 1):
            res += C_n_k(deg, i) * np.power(1 - t_i, deg - i) * np.power(t_i, i) * points[i]
        yield res


def draw_lines_custom(steps):
    for i in range(len(steps) - 1):
        draw_line_custom(steps[i].astype(int), steps[i + 1].astype(int))


def draw_points_custom(points):
    for p in points:
        pygame.draw.circle(sc, (255, 0, 0), (p[0], p[1]), 3)


def draw_line_custom(p1, p2):
    x1, y1 = p1.tolist()
    x2, y2 = p2.tolist()
    dx = x2 - x1
    dy = y2 - y1
    # pygame.draw.circle(sc, (255, 255, 0), (x1, y1), 3)
    # pygame.draw.circle(sc, (255, 255, 0), (x2, y2), 3)
    if dy == 0:
        if x1 > x2:
            b = x1
            x1 = x2
            x2 = b
        sc.set_at((x1, y1), BLACK)
        x = x1 + 1
        while x <= x2:
            sc.set_at((x, y1), BLACK)
            x += 1
    elif dx == 0:
        if y1 > y2:
            b = y1
            y1 = y2
            y2 = b
        sc.set_at((x1, y1), BLACK)
        y = y1 + 1
        while y <= y2:
            sc.set_at((x1, y), BLACK)
            y += 1
    elif math.fabs(dy / dx) < 1:
        if x1 > x2:
            b = (x1, y1)
            x1, y1 = x2, y2
            x2, y2 = b
        x = x1
        y = y1
        sc.set_at((x, y), BLACK)
        x += 1
        while x <= x2:
            if not ((y - 1 / 2 - y1) / (x - x1) < (y2 - y1) / (x2 - x1) < (y + 1 / 2 - y1) / (x - x1)):
                y += int(np.sign((y2 - y1) / (x2 - x1)))
            sc.set_at((x, y), BLACK)
            x += 1
    else:
        if y1 > y2:
            b = (x1, y1)
            x1, y1 = x2, y2
            x2, y2 = b
        x = x1
        y = y1
        sc.set_at((x, y), BLACK)
        y += 1
        while y <= y2:
            # if not ((y - y1) / (x + 1 / 2 - x1) < (y2 - y1) / (x2 - x1) < (y - y1) / (x - 1 / 2 - x1)):
            if not ((x - 1 / 2 - x1) / (y - y1) < (x2 - x1) / (y2 - y1) < (x + 1 / 2 - x1) / (y - y1)):
                x += int(np.sign((y2 - y1) / (x2 - x1)))
            sc.set_at((x, y), BLACK)
            y += 1


while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                print("Rand line")
                sc.fill(WHITE)
                rand_bezier()
                pygame.display.update()
            elif event.key == pygame.K_2:
                print("face_points")
                sc.fill(WHITE)
                face_points = get_face_points("low.txt")
                draw_iter_bezier(face_points, 5, 3, True)
                pygame.display.update()
            elif event.key == pygame.K_w:
                print("face_points")
                sc.fill(WHITE)
                face_points = get_face_points("mid.txt")
                draw_iter_bezier(face_points, 5, 3, True)
                pygame.display.update()
            elif event.key == pygame.K_s:
                print("face_points")
                sc.fill(WHITE)
                face_points = get_face_points("high.txt")
                draw_iter_bezier(face_points, 5, 3, True)
                pygame.display.update()
            elif event.key == pygame.K_3:
                print("frog")
                sc.fill(WHITE)
                face_points = 20*(20 - get_face_points("frog.txt"))
                draw_iter_bezier(face_points, 5, 3, True)
                pygame.display.update()
            elif event.key == pygame.K_4:
                print("mickey")
                sc.fill(WHITE)
                face_points = 20*(10 - get_face_points("mickey.txt"))
                draw_iter_bezier(face_points, 5, 3, False)
                face_points = 20 * (10 - get_face_points("m_eyes_1.txt"))
                draw_iter_bezier(face_points, 5, 3, True)
                face_points = 20 * (10 - get_face_points("m_eyes_2.txt"))
                draw_iter_bezier(face_points, 5, 3, True)
                face_points = 20 * (10 - get_face_points("m_mouth.txt"))
                draw_iter_bezier(face_points, 5, 3, False)
                face_points = 20 * (10 - get_face_points("m_nose.txt"))
                draw_iter_bezier(face_points, 5, 3, True)
                pygame.display.update()
            elif event.key == pygame.K_5:
                print("cat")
                sc.fill(WHITE)
                face_points = 20*(15 - get_face_points("cat_easy.txt"))
                draw_iter_bezier(face_points, 5, 3, True)
                pygame.display.update()
    clock.tick(FPS)
