import numpy as np
import cv2 as cv

video_file = 'chessboard.avi'
K = np.array([[1436.488, 0, 929.57],
              [0, 1438.959, 520.83],
              [0, 0, 1.1626]], dtype=np.float32)
dist_coeff = np.array([-0.2852754904152874, 0.1016466459919075,
                       -0.0004420196146339175, 0.0001149909868437517,
                       -0.01803978785585194])
board_pattern = (8, 6)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

lego_body = board_cellsize * np.array([
    [4, 2, 0], [6, 2, 0], [6, 4, 0], [4, 4, 0],
    [4, 2, -0.5], [6, 2, -0.5], [6, 4, -0.5], [4, 4, -0.5],
], dtype=np.float32)

lego_body_lines = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

nubs = []
for dx in [0.5, 1.5]:
    for dy in [0.5, 1.5]:
        center = board_cellsize * np.array([4 + dx, 2 + dy, -0.6])
        radius = board_cellsize * 0.2
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            nubs.append([x, y, z])
nubs_arr = np.array(nubs, dtype=np.float32)

obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

def draw_lego_block(img, rvec, tvec):
    projected, _ = cv.projectPoints(lego_body, rvec, tvec, K, dist_coeff)
    pts = projected.reshape(-1, 2).astype(int)

    cv.fillConvexPoly(img, pts[4:8], (0, 140, 255))
    cv.fillConvexPoly(img, np.array([pts[3], pts[2], pts[6], pts[7]]), (0, 140, 255))
    cv.fillConvexPoly(img, np.array([pts[0], pts[1], pts[5], pts[4]]), (0, 140, 255))
    cv.fillConvexPoly(img, np.array([pts[1], pts[2], pts[6], pts[5]]), (0, 140, 255))
    cv.fillConvexPoly(img, np.array([pts[0], pts[3], pts[7], pts[4]]), (0, 140, 255))

    for i, j in lego_body_lines:
        cv.line(img, pts[i], pts[j], (0, 0, 0), 2)

    projected_nubs, _ = cv.projectPoints(nubs_arr, rvec, tvec, K, dist_coeff)
    projected_nubs = projected_nubs.reshape(-1, 2).astype(int)
    for i in range(0, len(projected_nubs), 8):
        cv.fillConvexPoly(img, projected_nubs[i:i+8], (0, 140, 255))
        cv.polylines(img, [projected_nubs[i:i+8]], True, (0, 0, 0), 2)

video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

while True:
    valid, img = video.read()
    if not valid:
        break

    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)
        draw_lego_block(img, rvec, tvec)
        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    cv.imshow('Pose Estimation (LEGO Block)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27:
        break

video.release()
cv.destroyAllWindows()
