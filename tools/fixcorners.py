# -*- coding: utf-8 -*-
"""Разметка углов листа мышью — вспомогательный инструмент (в сдачу не входит).

Запуск:  python tools/fixcorners.py                      # все кадры из data/, разметка data/corners.json
         python tools/fixcorners.py data/02.jpg data/04.jpg

Показывает кадр и серую подсказку сканера; рядом с курсором — лупа ×4, чтобы попадать в угол
с точностью до пикселя. Клик рядом с уже поставленной точкой (ближе 25 px) — переставить её.

Клавиши:
  Enter / Space — принять (свои клики, а если их нет — серую подсказку) и сохранить;
  k — оставить то, что уже размечено в corners.json;  s — пропустить кадр;
  r — сбросить клики;  Backspace — убрать последний;  n — «листа в кадре нет» (null);
  d — спрятать/показать подсказку;  q / Esc — выход.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import docscan as ds  # noqa: E402

MAX_W, MAX_H = 1150, 800          # окно не больше экрана ноутбука
LOUPE, ZOOM = 190, 4.0            # панель лупы и увеличение


def view_scale(shape):
    h, w = shape[:2]
    return min(1.0, float(MAX_W) / w, float(MAX_H) / h)


def draw_text(img, s, org, scale=0.55, colour=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def annotate_one(img, name, draft, current):
    """Окно одного кадра → ('set', quad) | ('none', None) | ('keep', None) | ('skip', None) | ('quit', None)."""
    scale = view_scale(img.shape)
    h, w = img.shape[:2]
    W, H = int(round(w * scale)), int(round(h * scale))
    clicks, state = [], {"cursor": (W // 2, H // 2), "hint": True}
    title = "%s | Enter-prinyat  k-ostavit  r-sbros  n-lista net  s-propusk  d-podskazka" % name
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            state["cursor"] = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = x / scale, y / scale
            for i, p in enumerate(clicks):                       # клик рядом с точкой — переставить её
                if abs(p[0] - ix) + abs(p[1] - iy) < 25 / scale:
                    clicks[i] = (ix, iy)
                    return
            if len(clicks) < 4:
                clicks.append((ix, iy))
        elif event == cv2.EVENT_RBUTTONDOWN and clicks:
            clicks.pop()

    cv2.setMouseCallback(title, on_mouse)

    result = ("skip", None)
    while True:
        view = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        if state["hint"] and draft is not None:                  # серая подсказка сканера
            q = np.int32(np.round(np.float32(draft) * scale))
            cv2.polylines(view, [q], True, (140, 140, 140), 2, cv2.LINE_AA)
            for p in q:
                cv2.circle(view, (int(p[0]), int(p[1])), 4, (140, 140, 140), -1, cv2.LINE_AA)
        if current is not None:                                  # что уже записано в corners.json
            cv2.polylines(view, [np.int32(np.round(np.float32(current) * scale))], True, (60, 160, 255), 1, cv2.LINE_AA)
        for i, p in enumerate(clicks):
            pt = (int(round(p[0] * scale)), int(round(p[1] * scale)))
            cv2.circle(view, pt, 5, (0, 255, 255), -1, cv2.LINE_AA)
            draw_text(view, str(i + 1), (pt[0] + 8, pt[1] - 8), 0.6, (0, 255, 255))
        if len(clicks) == 4:
            cv2.polylines(view, [np.int32(np.round(np.float32(clicks) * scale))], True, (0, 255, 0), 2, cv2.LINE_AA)

        cx, cy = state["cursor"]                                 # лупа вокруг курсора
        ix, iy = cx / scale, cy / scale
        r = int(round(LOUPE / (2 * ZOOM)))
        x0, y0 = int(round(ix)) - r, int(round(iy)) - r
        patch = np.zeros((2 * r, 2 * r, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + 2 * r), min(h, y0 + 2 * r)
        if sx1 > sx0 and sy1 > sy0:
            patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
        if draft is not None and state["hint"]:
            cv2.polylines(patch, [np.int32(np.round(np.float32(draft) - [x0, y0]))], True, (140, 140, 140), 1, cv2.LINE_AA)
        for p in clicks:
            pc = (int(round(p[0] - x0)), int(round(p[1] - y0)))
            if 0 <= pc[0] < 2 * r and 0 <= pc[1] < 2 * r:
                cv2.circle(patch, pc, 2, (0, 255, 255), -1)
        big = cv2.resize(patch, (LOUPE, LOUPE), interpolation=cv2.INTER_NEAREST)
        c = LOUPE // 2
        cv2.line(big, (c - 14, c), (c + 14, c), (0, 0, 255), 1)
        cv2.line(big, (c, c - 14), (c, c + 14), (0, 0, 255), 1)
        cv2.rectangle(big, (0, 0), (LOUPE - 1, LOUPE - 1), (255, 255, 255), 1)
        if H >= LOUPE + 8 and W >= LOUPE + 8:
            view[4:4 + LOUPE, W - 4 - LOUPE:W - 4] = big
        draw_text(view, "%s  x=%.0f y=%.0f  klikov %d/4" % (name, ix, iy, len(clicks)), (8, 22), 0.6, (0, 255, 255))

        cv2.imshow(title, view)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            result = ("quit", None)
            break
        if key in (13, 10, 32):
            if len(clicks) == 4:
                result = ("set", ds.order_corners(np.float32(clicks)))
                break
            if draft is not None:
                result = ("set", ds.order_corners(np.float32(draft)))
                break
            if current is not None:
                result = ("keep", None)
                break
            draw_text(view, "nuzhno 4 klika ili 'n' (lista net)", (8, H - 40), 0.7, (0, 0, 255))
            cv2.imshow(title, view)
            cv2.waitKey(1000)
            continue
        if key == ord("k"):
            result = ("keep", None)
            break
        if key == ord("n"):
            result = ("none", None)
            break
        if key == ord("s"):
            result = ("skip", None)
            break
        if key == ord("r"):
            clicks.clear()
        if key in (8, 127) and clicks:
            clicks.pop()
        if key == ord("d"):
            state["hint"] = not state["hint"]
    cv2.destroyWindow(title)
    return result


def main(argv):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ann_path = os.path.join(root, "data", "corners.json")
    args = list(argv[1:])
    if "--ann" in args:
        i = args.index("--ann")
        ann_path = args[i + 1]
        del args[i:i + 2]
    if not args:
        d = os.path.join(root, "data")
        args = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    ann = ds.load_annotations(ann_path)
    saved = 0
    for p in args:
        name = os.path.basename(p)
        img = ds.imread(p)
        draft, info = ds.detect(img, "auto", long_side=640)
        current = np.float32(ann[name]) if ann.get(name) is not None else None
        print("%-10s podskazka: %-12s [v razmetke: %s]" % (
            name, "net" if draft is None else "ball %.2f" % info.get("score", 0),
            "net" if current is None else "est"), flush=True)
        what, quad = annotate_one(img, name, draft, current)
        if what == "set":
            ann[name] = ds._corners_list(quad)
        elif what == "none":
            ann[name] = None
        elif what == "quit":
            ds.save_annotations(ann_path, ann)
            break
        if what in ("set", "none"):
            saved += 1
            print("   -> %s" % (ann[name],), flush=True)
        elif what == "keep":
            print("   -> ostavleno: %s" % (ann.get(name),), flush=True)
        else:
            print("   -> propushcheno", flush=True)
        ds.save_annotations(ann_path, ann)
    print("gotovo: %d zapisej v %s (obnovleno %d)" % (len(ann), ann_path, saved))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
