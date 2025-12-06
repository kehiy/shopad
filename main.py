"""
Full pipeline to extract 10x10 handwritten digit cells from scanned
filled templates in folder `shopad_raw_scans/` and save normalized
28x28 images to processed/{0..9}/.

Drop-in script. Tune the constants (BORDER_PERCENT, MARGIN_PERCENT)
if needed for your specific template.
"""

import cv2
import numpy as np
import os
from glob import glob

# --- Parameters (tune if necessary) ---
RAW_DIR = "shopad_raw_scans"
OUT_DIR = "processed"
GRID_TARGET = 1000               # warp grid to GRID_TARGET x GRID_TARGET
EXPECTED_CELLS = 10
BORDER_PIXELS = 6                # inner trim from each cell after slicing (removes thin grid line)
LEFT_CROP_PERCENT = 0.16         # crop this fraction from left of cell to remove printed grades (tune 0.10-0.20)
MIN_GRID_DIM = 200               # ignore contours smaller than this
# -------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
for lbl in range(10):
    os.makedirs(f"{OUT_DIR}/{lbl}", exist_ok=True)

counters = {str(k): 1 for k in range(10)}

def resize_max(image, max_size):
    h, w = image.shape[:2]
    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def sort_points_grid(pts, rows=11, cols=11):
    """Sort intersection points into a rows x cols grid. pts = Nx2 array."""
    pts_sorted = pts[np.argsort(pts[:,1])]
    grouped = []
    row_h = (pts_sorted[-1,1] - pts_sorted[0,1]) / (rows - 1 + 1e-8)
    cur_row = [pts_sorted[0]]
    last_y = pts_sorted[0,1]
    for p in pts_sorted[1:]:
        if abs(p[1] - last_y) < row_h * 0.6:
            cur_row.append(p)
        else:
            grouped.append(np.array(cur_row))
            cur_row = [p]
            last_y = p[1]
    grouped.append(np.array(cur_row))
    if len(grouped) != rows:
        return None
    grid = []
    for row in grouped:
        row = row[np.argsort(row[:,0])]
        if len(row) != cols:
            return None
        grid.append(row)
    return np.array(grid, dtype=np.float32) 

def find_grid_by_morphology(gray):
    """
    Detect vertical and horizontal lines by morphological filtering.
    Return intersections (corner points) as Nx2 numpy array.
    """
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw

    horizontal = bw.copy()
    vertical = bw.copy()

    cols = horizontal.shape[1]
    horizontal_size = max(10, cols // 30)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horiz_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, horiz_kernel, iterations=1)

    rows = vertical.shape[0]
    vertical_size = max(10, rows // 30)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vert_kernel, iterations=1)
    vertical = cv2.dilate(vertical, vert_kernel, iterations=1)

    grid_lines = cv2.bitwise_and(horizontal, vertical)

    inter = cv2.bitwise_and(horizontal, vertical)

    cnts, _ = cv2.findContours(inter, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cx = x + w//2
        cy = y + h//2
        pts.append((cx, cy))
    pts = np.array(pts, dtype=np.float32)
    return pts, bw, grid_lines

def four_point_transform(image, pts):
    """Warp image to rectangle based on 4 corner points (tl,tr,br,bl)."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # tl
    rect[2] = pts[np.argmax(s)]    # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # tr
    rect[3] = pts[np.argmax(diff)] # bl
    return rect

def extract_cells_from_grid(warped_gray):
    """
    Given a corrected, top-down grayscale image of the grid (warped),
    try to find intersections to produce an (11 x 11) grid of points.
    Return list of cell images (row-major) and (row,col) coordinates.
    """
    pts, _, _ = find_grid_by_morphology(warped_gray)
    if pts is None or len(pts) < (EXPECTED_CELLS+1)*(EXPECTED_CELLS+1)*0.6:
        H, W = warped_gray.shape[:2]
        cell_h = H // EXPECTED_CELLS
        cell_w = W // EXPECTED_CELLS
        cells = []
        coords = []
        for r in range(EXPECTED_CELLS):
            for c in range(EXPECTED_CELLS):
                y1 = r * cell_h
                y2 = (r+1) * cell_h
                x1 = c * cell_w
                x2 = (c+1) * cell_w
                cells.append(warped_gray[y1:y2, x1:x2])
                coords.append((r,c))
        return cells, coords

    grid = sort_points_grid(pts, rows=EXPECTED_CELLS+1, cols=EXPECTED_CELLS+1)
    if grid is None:
        H, W = warped_gray.shape[:2]
        cell_h = H // EXPECTED_CELLS
        cell_w = W // EXPECTED_CELLS
        cells = []
        coords = []
        for r in range(EXPECTED_CELLS):
            for c in range(EXPECTED_CELLS):
                y1 = r * cell_h
                y2 = (r+1) * cell_h
                x1 = c * cell_w
                x2 = (c+1) * cell_w
                cells.append(warped_gray[y1:y2, x1:x2])
                coords.append((r,c))
        return cells, coords

    cells = []
    coords = []
    for r in range(EXPECTED_CELLS):
        for c in range(EXPECTED_CELLS):
            tl = grid[r, c]
            tr = grid[r, c+1]
            bl = grid[r+1, c]
            br = grid[r+1, c+1]

            min_x = int(min(tl[0], tr[0], bl[0], br[0]))
            max_x = int(max(tl[0], tr[0], bl[0], br[0]))
            min_y = int(min(tl[1], tr[1], bl[1], br[1]))
            max_y = int(max(tl[1], tr[1], bl[1], br[1]))

            dx = max(3, int((max_x-min_x)*0.02))
            dy = max(3, int((max_y-min_y)*0.02))
            x1 = min_x + dx
            x2 = max_x - dx
            y1 = min_y + dy
            y2 = max_y - dy

            H, W = warped_gray.shape[:2]
            x1 = max(0, min(W-1, x1))
            x2 = max(0, min(W, x2))
            y1 = max(0, min(H-1, y1))
            y2 = max(0, min(H, y2))
            if x2 - x1 <= 4 or y2 - y1 <= 4:
                cell_img = np.zeros((20,20), dtype=np.uint8)
            else:
                cell_img = warped_gray[y1:y2, x1:x2]
            cells.append(cell_img)
            coords.append((r,c))
    return cells, coords

def normalize_and_center(cell_img, final_size=28, inner_size=20):
    """
    Convert a grayscale cell image (digit + some noise) into a MNIST-like 28x28 image:
    - crop a left fraction to remove printed grade if needed
    - threshold using Otsu
    - remove small components touching borders
    - find largest connected component = digit
    - resize digit maintaining aspect to inner_size (<=20)
    - center into final_size x final_size canvas
    - ensure digit is white (255) on black background (0)
    """
    if len(cell_img.shape) == 3:
        cell = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        cell = cell_img.copy()

    H, W = cell.shape
    trim = BORDER_PIXELS
    if W > trim*4 and H > trim*4:
        cell = cell[trim:H-trim, trim:W-trim]

    left_crop = int(cell.shape[1] * LEFT_CROP_PERCENT)
    if left_crop > 0:
        cell = cell[:, left_crop:]

    cell = cv2.GaussianBlur(cell, (3,3), 0)

    _, th = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    Ht, Wt = th.shape
    cleaned = np.zeros_like(th)
    areas = []
    for lab in range(1, num_labels):
        x, y, w, h, area = stats[lab]
        touching_border = (x <= 1) or (y <= 1) or (x+w >= Wt-2) or (y+h >= Ht-2)
        if not touching_border and area > 30:
            cleaned[labels == lab] = 255
            areas.append((area, lab))
    if cleaned.sum() == 0:
        if num_labels > 1:
            largest = 1 + np.argmax([stats[i][4] for i in range(1, num_labels)])
            cleaned[labels == largest] = 255

    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        canvas = np.zeros((final_size, final_size), dtype=np.uint8)
        return canvas

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    digit = cleaned[y:y+h, x:x+w]

    h_d, w_d = digit.shape
    if h_d > 0 and w_d > 0:
        scale = min(inner_size / h_d, inner_size / w_d)
        new_w = max(1, int(w_d * scale))
        new_h = max(1, int(h_d * scale))
        digit_rs = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        digit_rs = cv2.resize(digit, (inner_size, inner_size), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((final_size, final_size), dtype=np.uint8)
    x_offset = (final_size - digit_rs.shape[1]) // 2
    y_offset = (final_size - digit_rs.shape[0]) // 2
    canvas[y_offset:y_offset+digit_rs.shape[0], x_offset:x_offset+digit_rs.shape[1]] = digit_rs

    canvas = cv2.medianBlur(canvas, 3)
    return canvas

def process_one_image(path):
    global counters
    print("Processing:", path)
    img = cv2.imread(path)
    if img is None:
        print("Failed to read:", path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_inv = 255 - bw
    cnts, _ = cv2.findContours(bw_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page_cnt = None
    max_area = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area > max_area and w > MIN_GRID_DIM and h > MIN_GRID_DIM:
            max_area = area
            page_cnt = c

    if page_cnt is None:
        print("Grid/page contour not found - skipping:", path)
        return

    peri = cv2.arcLength(page_cnt, True)
    approx = cv2.approxPolyDP(page_cnt, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4,2).astype(np.float32)
        warped = four_point_transform(img, pts)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    else:
        rect = cv2.minAreaRect(page_cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        warped = four_point_transform(img, box.astype(np.float32))
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    warped_gray = resize_max(warped_gray, GRID_TARGET)
    warped_gray = cv2.resize(warped_gray, (GRID_TARGET, GRID_TARGET), interpolation=cv2.INTER_AREA)

    cells, coords = extract_cells_from_grid(warped_gray)

    for (cell_img, (r,c)) in zip(cells, coords):
        h, w = cell_img.shape[:2]
        if h > BORDER_PIXELS*2 and w > BORDER_PIXELS*2:
            cell_trim = cell_img[BORDER_PIXELS:h-BORDER_PIXELS, BORDER_PIXELS:w-BORDER_PIXELS]
        else:
            cell_trim = cell_img

        out_img = normalize_and_center(cell_trim, final_size=28, inner_size=20)

        label = str(c)
        count = counters[label]
        filename = f"{OUT_DIR}/{label}/{count:06}.png"
        cv2.imwrite(filename, out_img)
        counters[label] += 1

def main():
    paths = sorted(glob(os.path.join(RAW_DIR, "*.*")))
    if len(paths) == 0:
        print("No files found in", RAW_DIR)
        return
    for p in paths:
        try:
            process_one_image(p)
        except Exception as e:
            print("Error processing", p, ":", e)

    print("Done. Totals:")
    for k in sorted(counters.keys()):
        print(k, counters[k]-1)

if __name__ == "__main__":
    main()
