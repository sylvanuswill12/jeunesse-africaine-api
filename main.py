"""
Backend — Jeunesse Africaine en Action
=======================================
Fournit un point d'API unique qui :
  1. reçoit la photo de l'utilisateur,
  2. détecte le visage et calcule un recadrage centré (visage + buste),
  3. redimensionne/recadre la photo pour qu'elle remplisse EXACTEMENT le
     cadre (comme un "cover" CSS) — garantit qu'il n'y a jamais ni zone
     blanche, ni débordement, quelle que soit la photo envoyée,
  4. insère la photo dans le cadre exact de l'affiche officielle, avec les
     mêmes coins arrondis que le cadre imprimé,
  5. renvoie l'affiche personnalisée en PNG haute définition.

Conforme au cahier des charges (§4.2 et §4.3) :
  - La photo d'origine n'est JAMAIS écrite sur disque : tout est traité en
    mémoire et supprimé dès la réponse envoyée (RGPD).
  - Aucune base de données.

Note de conception : une version précédente supprimait aussi l'arrière-plan
de la photo (modèle IA U^2-Net/rembg) pour un effet "détouré". Ça a été
retiré : les zones où l'arrière-plan était supprimé devenaient transparentes,
ce qui laissait apparaître le fond blanc de l'affiche à travers (visible
comme des "taches blanches" dans le cadre) — en plus de consommer beaucoup
de mémoire (cause de plantages "out of memory" sur l'instance gratuite).
Le recadrage "cover" utilisé ici garantit mathématiquement une couverture
à 100% du cadre, sans aucune zone transparente possible.
"""

import gc
import io
import logging

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jeunesse-africaine-api")

MAX_INPUT_DIMENSION = 1280

POSTER_PATH = "poster_template.png"
FRAME_X, FRAME_Y, FRAME_W, FRAME_H = 741, 169, 473, 652
FRAME_INSET = 16
FRAME_RADIUS = 58
FACE_VERTICAL_BIAS = 0.10
SUBJECT_ZOOM = 1.08

app = FastAPI(
    title="Jeunesse Africaine en Action — API de composition d'affiche",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_poster_cache = None
_face_cascade = None


def get_poster():
    global _poster_cache
    if _poster_cache is None:
        _poster_cache = Image.open(POSTER_PATH).convert("RGBA")
    return _poster_cache.copy()


def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        import cv2
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


def detect_face_box(rgb_image):
    import cv2
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    cascade = get_face_cascade()
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def apply_rounded_frame_mask(subject):
    w, h = subject.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        [FRAME_INSET, FRAME_INSET, w - FRAME_INSET, h - FRAME_INSET],
        radius=FRAME_RADIUS,
        fill=255,
    )
    subject.putalpha(mask)
    return subject


def smart_crop_and_fit(photo_rgba, original_rgb):
    face = detect_face_box(original_rgb)
    iw, ih = photo_rgba.size

    base_scale = max(FRAME_W / iw, FRAME_H / ih) * SUBJECT_ZOOM
    new_w, new_h = int(iw * base_scale) + 1, int(ih * base_scale) + 1
    resized = photo_rgba.resize((new_w, new_h), Image.LANCZOS)

    if face is not None:
        fx, fy, fw, fh = face
        face_cx = (fx + fw / 2) * base_scale
        face_cy = (fy + fh / 2) * base_scale
    else:
        face_cx = new_w / 2
        face_cy = new_h / 2

    face_cy -= FRAME_H * FACE_VERTICAL_BIAS

    left = int(max(0, min(new_w - FRAME_W, face_cx - FRAME_W / 2)))
    top = int(max(0, min(new_h - FRAME_H, face_cy - FRAME_H / 2)))

    cropped = resized.crop((left, top, left + FRAME_W, top + FRAME_H))
    return cropped


def compose_poster(subject_crop):
    poster = get_poster()
    poster.alpha_composite(subject_crop, dest=(FRAME_X, FRAME_Y))
    return poster


def downscale_if_needed(img):
    w, h = img.size
    longest = max(w, h)
    if longest <= MAX_INPUT_DIMENSION:
        return img
    scale = MAX_INPUT_DIMENSION / longest
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/compose")
async def compose(photo: UploadFile = File(...)):
    if not photo.content_type or not photo.content_type.startswith("image/"):
        raise HTTPException(400, "Le fichier envoyé doit être une image.")

    raw_bytes = await photo.read()

    try:
        original = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Image illisible ou corrompue.")

    del raw_bytes
    original = downscale_if_needed(original)
    original_np = np.array(original)
    photo_rgba = original.convert("RGBA")

    cropped = smart_crop_and_fit(photo_rgba, original_np)
    del photo_rgba, original_np, original
    cropped = apply_rounded_frame_mask(cropped)
    final_poster = compose_poster(cropped)
    del cropped

    gc.collect()

    buffer = io.BytesIO()
    final_poster.convert("RGB").save(buffer, format="PNG")
    buffer.seek(0)
    del final_poster

    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={
            "Content-Disposition": 'inline; filename="jeunesse-africaine-en-action.png"'
        },
  )
  
