import cv2
import numpy as np
from pathlib import Path
import json

# ====== SET YOUR IMAGE PATH HERE ======
IN_PATH = Path("/Users/andredelacruz/Documents/GitHub/cpchess/test.png")
# ======================================

OUT_DIR = IN_PATH.parent
OUT_GRID = OUT_DIR / "inner_square_with_grid.png"

# ---------------- TUNABLES ----------------
LOWER_BLUE = np.array([90, 60, 40], dtype=np.uint8)
LOWER_BLUE_REMOVAL = np.array([90, 100, 100], dtype=np.uint8)
UPPER_BLUE = np.array([120, 255, 255], dtype=np.uint8)
BOARD_SIDE = 800
EDGE_MARGIN_FRAC = 0.05
CLOSE_K = 5
MIN_INNER_AREA_FRAC = 0.02
AR_TOL = 2
RECTANGULARITY_MIN = 0.75
LOWER_RED1 = np.array([0, 100, 80], dtype=np.uint8)
UPPER_RED1 = np.array([10, 255, 255], dtype=np.uint8)
LOWER_RED2 = np.array([160, 100, 80], dtype=np.uint8)
UPPER_RED2 = np.array([179, 255, 255], dtype=np.uint8)

# Placeholder ranges for other pieces
# TODO: Adjust these later with real colors
LOWER_KNIGHT, UPPER_KNIGHT = None, None
LOWER_ROOK,   UPPER_ROOK   = None, None
LOWER_BISHOP, UPPER_BISHOP = None, None
LOWER_QUEEN,  UPPER_QUEEN  = None, None
# ------------------------------------------
def _odd(k): 
    return int(k) + 1 - (int(k) % 2)  # make odd

def estimate_grid_positions(crop, n=9, win_frac=0.045, smooth_frac=0.035):
    """
    Return (verticals, horizontals) 1D indices for the n grid lines in each direction.
    Uses Sobel projections + local peak search around the expected equal-spaced locations.
    """
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Light blur so tiny piece edges don't dominate
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Sobel gradients (vertical lines => strong d/dx; horizontal lines => strong d/dy)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Column/row "gridness" scores
    vert_score = np.mean(np.abs(gx), axis=0)   # length w
    horiz_score = np.mean(np.abs(gy), axis=1)  # length h

    # Smooth the 1D signals (kernel size proportional to size, and odd)
    kx = _odd(max(7, int(w * smooth_frac)))
    ky = _odd(max(7, int(h * smooth_frac)))
    vert_score = cv2.GaussianBlur(vert_score.reshape(1,-1), (1, kx), 0).ravel()
    horiz_score = cv2.GaussianBlur(horiz_score.reshape(-1,1), (ky, 1), 0).ravel()

    # Start from equal spacing, then snap each line to the strongest local peak nearby
    vx = np.linspace(0, w-1, n).astype(int)
    vy = np.linspace(0, h-1, n).astype(int)

    wx = max(3, int(w * win_frac))   # +/- window in which to search for a better peak
    wy = max(3, int(h * win_frac))

    def snap_peaks(starts, score, win, limit):
        snapped = []
        for s in starts:
            a = max(0, s - win)
            b = min(limit-1, s + win)
            local = score[a:b+1]
            j = np.argmax(local)
            snapped.append(int(a + j))
        # enforce monotonic order and unique indices
        snapped = np.clip(np.array(snapped, dtype=int), 0, limit-1)
        snapped = np.maximum.accumulate(snapped)           # monotonic non-decreasing
        # if any duplicates (rare), spread them minimally
        for i in range(1, len(snapped)):
            if snapped[i] <= snapped[i-1]:
                snapped[i] = min(limit-1, snapped[i-1] + 1)
        return snapped.tolist()

    verticals   = snap_peaks(vx, vert_score, wx, w)
    horizontals = snap_peaks(vy, horiz_score, wy, h)

    # Sanity: require first/last near edges; if not, fall back to exact linspace
    if verticals[0] > w*0.05 or verticals[-1] < w*0.95 or \
       horizontals[0] > h*0.05 or horizontals[-1] < h*0.95:
        verticals   = np.linspace(0, w-1, n).astype(int).tolist()
        horizontals = np.linspace(0, h-1, n).astype(int).tolist()

    return verticals, horizontals

def order_corners(pts):
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def find_largest_blue_quad(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000: 
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
            best_area = area
            best = approx
    return best  

def rectangularity(cnt):
    a = max(cv2.contourArea(cnt), 1.0)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box_a = max(cv2.contourArea(box.astype(np.int32)), 1.0)
    return a / box_a

def snap_rect_to_square(rect):
    (cx, cy), (w, h), ang = rect
    side = max(w, h)
    return ((cx, cy), (side, side), ang)

def rect_to_points(rect):
    box = cv2.boxPoints(rect)
    return order_corners(box)

def main():
    img = cv2.imread(str(IN_PATH))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {IN_PATH}")

    # Outer square detection and warp
    outer_quad = find_largest_blue_quad(img)
    if outer_quad is None:
        raise RuntimeError("Outer blue square not found.")
    outer_quad = order_corners(outer_quad.reshape(-1,2))

    dst = np.array([[0,0],[BOARD_SIDE-1,0],[BOARD_SIDE-1,BOARD_SIDE-1],[0,BOARD_SIDE-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(outer_quad, dst)
    warp = cv2.warpPerspective(img, H, (BOARD_SIDE, BOARD_SIDE))

    # Mask for inner square
    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    margin = int(BOARD_SIDE * EDGE_MARGIN_FRAC)
    center_mask = np.zeros_like(mask)
    cv2.rectangle(center_mask, (margin, margin), (BOARD_SIDE - margin, BOARD_SIDE - margin), 255, -1)
    mask_inner = cv2.bitwise_and(mask, center_mask)

    # Morphological cleanup
    k = max(3, CLOSE_K | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(mask_inner, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Candidate filtering
    Hh, Ww = closed.shape
    min_area = Hh * Ww * MIN_INNER_AREA_FRAC
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        hull = cv2.convexHull(c)
        rect = cv2.minAreaRect(hull)
        (w, h) = rect[1]
        if w == 0 or h == 0:
            continue
        ar = max(w, h) / (min(w, h) + 1e-6)
        if ar > AR_TOL:
            continue
        if rectangularity(hull) < RECTANGULARITY_MIN:
            continue
        candidates.append((cv2.contourArea(hull), rect))

    if not candidates:
        raise RuntimeError("Inner blue square not found.")

    # Pick the smallest candidate (inner square)
    candidates.sort(key=lambda x: x[0])
    best_rect_sq = snap_rect_to_square(candidates[0][1])
    inner_pts = rect_to_points(best_rect_sq)

    # Crop to inner square
    side = int(max(
        np.linalg.norm(inner_pts[0] - inner_pts[1]),
        np.linalg.norm(inner_pts[1] - inner_pts[2]),
        np.linalg.norm(inner_pts[2] - inner_pts[3]),
        np.linalg.norm(inner_pts[3] - inner_pts[0]),
    ))
    side = max(side, 200)
    dst2 = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype=np.float32)
    H2 = cv2.getPerspectiveTransform(inner_pts, dst2)
    crop = cv2.warpPerspective(warp, H2, (side, side))

    # ===== Remove all shades of blue =====
    # ===== Remove all shades of blue and crop =====
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv_crop, LOWER_BLUE_REMOVAL, UPPER_BLUE)

    # Invert mask to keep non-blue regions
    non_blue_mask = cv2.bitwise_not(blue_mask)

    # Apply mask
    non_blue = cv2.bitwise_and(crop, crop, mask=non_blue_mask)

    # Find bounding box of non-blue pixels
    coords = cv2.findNonZero(non_blue_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        crop = non_blue[y:y+h, x:x+w]
    else:
        raise RuntimeError("No non-blue region found after masking.")

    # ===== Robust grid estimation (projection-based, piece-robust) =====
    verticals, horizontals = estimate_grid_positions(crop, n=9, win_frac=0.045, smooth_frac=0.035)


    def dedup_and_fill(vals, size, max_len):
        vals = sorted(set(vals))
        # deduplicate close values
        filtered = []
        for v in vals:
            if not filtered or abs(v - filtered[-1]) > 10:
                filtered.append(v)

        # if we have fewer than expected → interpolate
        if len(filtered) < size:
            start, end = 0, max_len - 1
            filtered = np.linspace(start, end, size, dtype=int).tolist()

        # if too many, thin them evenly
        if len(filtered) > size:
            idxs = np.linspace(0, len(filtered)-1, size, dtype=int)
            filtered = [filtered[i] for i in idxs]

        return filtered
        # ---- PIECE DETECTION CONSTANTS (red pawn implemented; others placeholders) ----
        
    LOWER_RED1 = np.array([0, 100, 80],  dtype=np.uint8)
    UPPER_RED1 = np.array([10, 255, 255], dtype=np.uint8)
    LOWER_RED2 = np.array([160, 100, 80], dtype=np.uint8)
    UPPER_RED2 = np.array([179, 255, 255], dtype=np.uint8)

    PIECE_MIN_FRAC = 0.03   # >=3% of square pixels in color mask => treat as present
    BRIGHTNESS_SPLIT = 140  # V-channel mean to decide White vs Black piece

    def detect_red_pawn(roi_bgr):
        """Return ('P','White'/'Black') if red pawn present, else (None,None)."""
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
        mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
        mask  = cv2.bitwise_or(mask1, mask2)

        area = roi_bgr.shape[0] * roi_bgr.shape[1]
        if area == 0:
            return (None, None)

        count = cv2.countNonZero(mask)
        if count >= area * PIECE_MIN_FRAC:
            vals = hsv[:, :, 2][mask > 0]
            color = "White" if (np.mean(vals) if vals.size else 0) >= BRIGHTNESS_SPLIT else "Black"
            return ('P', color)
        return (None, None)

    # ---- BUILD SQUARES, LABEL, AND OUTPUT BOARD LIST ----
    verticals   = sorted(verticals)
    horizontals = sorted(horizontals)
    h, w = crop.shape[:2]
    grid = crop.copy()

    files_nums   = [str(i) for i in range(1, 9)]                 # 1..8 (left -> right)
    ranks_letters = [chr(ord('A') + i) for i in range(8)]        # A..H (top -> bottom)

    # optional: store coords + piece info for later use
    square_info = {}

    # your requested list: [ ((Piece, Color), number, letter), ... ]  (64 items)
    board_list = []

    # helper for pretty text
    PIECE_NAME = {'P': 'pawn', 'N': 'knight', 'B': 'bishop', 'R': 'rook', 'Q': 'queen', 'K': 'king'}

    for row in range(8):        # A..H (top -> bottom)
        for col in range(8):    # 1..8 (left -> right)
            x1, x2 = int(verticals[col]), int(verticals[col+1])
            y1, y2 = int(horizontals[row]), int(horizontals[row+1])

            roi = crop[y1:y2, x1:x2]

            # --- detect pieces ---
            piece_code, piece_color = detect_red_pawn(roi)  # 'P' or None, 'White'/'Black' or None

            # TODO (placeholders): add color-coded detectors for N, B, R, Q later
            # piece_code, piece_color = detect_knight(roi) or existing result ...
            # piece_code, piece_color = detect_bishop(roi) ...
            # piece_code, piece_color = detect_rook(roi) ...
            # piece_code, piece_color = detect_queen(roi) ...

            # --- store metadata ---
            square_name = f"{ranks_letters[row]}{files_nums[col]}"
            # square_info[square_name] = {
            #     "coords": [x1, y1, x2, y2],
            #     "piece": {
            #         "code": piece_code,                  # 'P' or None
            #         "type": PIECE_NAME.get(piece_code),  # 'pawn' or None
            #         "color": piece_color                 # 'White'/'Black' or None
            #     }
            # }
            # ((Piece, Color), number, letter)  -->  (None, number, letter) if empty
            first = (piece_code, piece_color) if piece_code else None
            board_list.append((first, files_nums[col], ranks_letters[row]))

            # --- draw labels (two lines) ---
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(grid, square_name, (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            if piece_code:
                piece_label = f"{piece_color} {PIECE_NAME[piece_code]}"
                cv2.putText(grid, piece_label, (cx - 35, cy + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)

            # --- append to requested board list ---
            # ((Piece, Color), number, letter)

    # Save labeled image
    cv2.imwrite(str(OUT_GRID), grid)


    # Save the 64-tuple list in a Python-friendly text file and also print it
    OUT_BOARD_LIST = OUT_DIR / "board_list.txt"
    with open(OUT_BOARD_LIST, "w") as f:
        f.write(repr(board_list))

    print("Board list (64 tuples):")
    print(board_list)
    print(f"\nSaved labeled board image: {OUT_GRID}")
    print(f"Saved tuple list          : {OUT_BOARD_LIST}")

    h, w = crop.shape[:2]
    verticals   = dedup_and_fill(verticals,   9, w)
    horizontals = dedup_and_fill(horizontals, 9, h)

    # draw grid
    h, w = crop.shape[:2]
    verticals   = dedup_and_fill(verticals,   9, w)
    horizontals = dedup_and_fill(horizontals, 9, h)

    verticals = sorted(verticals)
    horizontals = sorted(horizontals)

    # build chessboard squares
    files = [str(i) for i in range(1,9)]   # left to right
    ranks = [chr(ord('A')+i) for i in range(8)]  # A..H top to bottom

    grid = crop.copy()
    square_info = {}


    for row in range(8):       # rank (A..H, top→bottom)
        for col in range(8):   # file (1..8, left→right)
            x1, x2 = verticals[col], verticals[col+1]
            y1, y2 = horizontals[row], horizontals[row+1]

            # square coordinates
            roi = crop[y1:y2, x1:x2]
            square_name = f"{ranks[row]}{files[col]}"

            # save metadata
            piece_entry = {"code": piece_code, "type": PIECE_NAME.get(piece_code), "color": piece_color} \
              if piece_code else None
            
            square_info[square_name] = {
                "coords": [x1, y1, x2, y2],
                "piece": piece_entry
            }
            # draw rectangle
            cv2.rectangle(grid, (x1,y1), (x2,y2), (0,0,255), 1)

            # put text in center
            cx, cy = (x1+x2)//2, (y1+y2)//2

                        # --- Piece detection by color coding ---
            piece_type = None
            piece_color = None

            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Red mask (Pawn)
            mask_red1 = cv2.inRange(hsv_roi, LOWER_RED1, UPPER_RED1)
            mask_red2 = cv2.inRange(hsv_roi, LOWER_RED2, UPPER_RED2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            red_pixels = cv2.countNonZero(mask_red)
            area = roi.shape[0] * roi.shape[1]

            if red_pixels > area * 0.05:  # ≥5% of square is red
                piece_type = "pawn"

                # Piece color (white/black) → based on brightness
                vals = hsv_roi[:,:,2][mask_red > 0]  # V channel where piece is
                if np.mean(vals) > 128:
                    piece_color = "white"
                else:
                    piece_color = "black"

            # TODO: Add knight, rook, bishop, queen detections here later

            # Save metadata
            square_info[square_name] = {
                "coords": [int(x1), int(y1), int(x2), int(y2)],
                "piece": {
                    "type": piece_type if piece_type else "empty",
                    "color": piece_color if piece_color else None
                }
            }

            # Draw label on grid
            label = square_name

            cv2.putText(grid, square_name, (cx-20, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,100), 1, cv2.LINE_AA)

            # Draw piece info (if exists) just below the name
            if piece_type:
                piece_label = f"{piece_color} {piece_type}"
                cv2.putText(grid, piece_label, (cx-35, cy+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


    # Save annotated grid
    cv2.imwrite(str(OUT_GRID), grid)

    print("Saved:")
    print(f"  Final labeled chessboard image : {OUT_GRID}")

    # Save coords back in original image space
    H_inv = np.linalg.inv(H)
    inner_pts_h = np.hstack([inner_pts, np.ones((4,1), dtype=np.float32)])
    inner_orig = (H_inv @ inner_pts_h.T).T
    inner_orig = (inner_orig[:, :2] / inner_orig[:, 2:3]).astype(float)

if __name__ == "__main__":
    main()
