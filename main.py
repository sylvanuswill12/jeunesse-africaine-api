"""
Backend IA — Jeunesse Africaine en Action
==========================================
Fournit un point d'API unique qui :
  1. reçoit la photo de l'utilisateur,
  2. détecte le visage et calcule un recadrage centré (visage + buste),
  3. supprime l'arrière-plan (modèle U^2-Net via la librairie `rembg`),
  4. insère le sujet détouré dans le cadre exact de l'affiche officielle,
  5. renvoie l'affiche personnalisée en PNG haute définition.

Conforme au cahier des charges (§4.2 et §4.3) :
  - La photo d'origine n'est JAMAIS écrite sur disque : tout est traité en mémoire
    et supprimé dès la réponse envoyée (RGPD).
  - Aucune base de données.
"""

import gc
import io
import logging
import os

# Réduit l'empreinte mémoire du moteur d'inférence IA (chaque thread
# supplémentaire alloue ses propres buffers) — important sur une instance
# à 512 Mo de RAM. À placer avant tout import d'onnxruntime/rembg.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("ORT_NUM_THREADS", "1")

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jeunesse-africaine-api")

# Limite anti-surcharge mémoire : les photos de téléphone modernes (12+
# mégapixels) font exploser la RAM du modèle IA sur un plan gratuit (512 Mo).
# On downscale systématiquement avant tout traitement — largement suffisant
# puisque la photo finale n'occupe qu'une petite portion de l'affiche.
MAX_INPUT_DIMENSION = 1280

# ----------------------------------------------------------------------------
# Configuration — coordonnées exactes du cadre, mesurées sur l'affiche officielle
# (image 1254x1254 px fournie par l'ONG-AIL4C)
# ----------------------------------------------------------------------------
POSTER_PATH = "poster_template.png"
FRAME_X, FRAME_Y, FRAME_W, FRAME_H = 741, 169, 473, 652
# Le cadre imprimé a des coins très arrondis (~70-75px mesurés sur l'affiche).
# On applique le même arrondi + une petite marge pour ne jamais déborder sur
# le trait de pinceau orange/vert du cadre.
FRAME_INSET = 9
FRAME_RADIUS = 62
FACE_VERTICAL_BIAS = 0.10  # laisse un peu plus d'espace sous le visage pour le buste
SUBJECT_ZOOM = 1.08        # léger zoom pour un cadrage plus serré et flatteur

app = FastAPI(
    title="Jeunesse Africaine en Action — API de composition d'affiche",
    version="1.0.0",
)

# En production, restreindre allow_origins au domaine réel du site (Netlify, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_poster_cache: Image.Image | None = None
_face_cascade = None
_rembg_session = None


def get_poster() -> Image.Image:
    global _poster_cache
    if _poster_cache is None:
        _poster_cache = Image.open(POSTER_PATH).convert("RGBA")
    return _poster_cache.copy()


def get_face_cascade():
    """Détecteur de visage léger (Haar cascade, livré avec opencv-python)."""
    global _face_cascade
    if _face_cascade is None:
        import cv2

        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


def get_rembg_session():
    """Session rembg (modèle U^2-Net), chargée une seule fois au démarrage."""
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session

        _rembg_session = new_session("u2net")
    return _rembg_session


def detect_face_box(rgb_image: np.ndarray):
    """Retourne (x, y, w, h) du plus grand visage détecté, ou None."""
    import cv2

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    cascade = get_face_cascade()
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    # on garde le plus grand visage détecté (sujet principal)
    return max(faces, key=lambda f: f[2] * f[3])


def remove_background(rgba_or_rgb_bytes: bytes) -> Image.Image:
    """Supprime l'arrière-plan et renvoie une image RGBA avec canal alpha."""
    from rembg import remove

    session = get_rembg_session()
    output_bytes = remove(rgba_or_rgb_bytes, session=session)
    return Image.open(io.BytesIO(output_bytes)).convert("RGBA")


def apply_rounded_frame_mask(subject: Image.Image) -> Image.Image:
    """
    Applique un masque à coins très arrondis (identique au cadre imprimé) sur
    le sujet déjà recadré à la taille du cadre, avec une petite marge pour ne
    jamais chevaucher le trait de pinceau du cadre.
    """
    w, h = subject.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        [FRAME_INSET, FRAME_INSET, w - FRAME_INSET, h - FRAME_INSET],
        radius=FRAME_RADIUS,
        fill=255,
    )
    # combine avec le canal alpha existant (transparence déjà retirée par rembg)
    r, g, b, a = subject.split()
    combined_alpha = Image.composite(a, Image.new("L", (w, h), 0), mask)
    subject.putalpha(combined_alpha)
    return subject


def smart_crop_and_fit(subject: Image.Image, original_rgb: np.ndarray) -> Image.Image:
    """
    Recadre et redimensionne le sujet (déjà détouré) pour remplir exactement
    le cadre FRAME_W x FRAME_H, en centrant sur le visage détecté si possible.
    """
    face = detect_face_box(original_rgb)
    iw, ih = subject.size

    # échelle "cover" : l'image remplit entièrement le cadre
    base_scale = max(FRAME_W / iw, FRAME_H / ih) * SUBJECT_ZOOM
    new_w, new_h = int(iw * base_scale), int(ih * base_scale)
    resized = subject.resize((new_w, new_h), Image.LANCZOS)

    if face is not None:
        fx, fy, fw, fh = face
        face_cx = (fx + fw / 2) * base_scale
        face_cy = (fy + fh / 2) * base_scale
    else:
        # pas de visage détecté : on centre simplement l'image (fallback)
        face_cx = new_w / 2
        face_cy = new_h / 2 - FRAME_H * FACE_VERTICAL_BIAS

    # décalage vertical pour laisser de la place au buste sous le visage
    face_cy -= FRAME_H * FACE_VERTICAL_BIAS

    left = int(max(0, min(new_w - FRAME_W, face_cx - FRAME_W / 2)))
    top = int(max(0, min(new_h - FRAME_H, face_cy - FRAME_H / 2)))

    cropped = resized.crop((left, top, left + FRAME_W, top + FRAME_H))
    return cropped


def compose_poster(subject_crop: Image.Image) -> Image.Image:
    """Colle le sujet détouré dans le cadre de l'affiche officielle."""
    poster = get_poster()
    poster.alpha_composite(subject_crop, dest=(FRAME_X, FRAME_Y))
    return poster


def downscale_if_needed(img: Image.Image) -> Image.Image:
    """Réduit une photo trop grande AVANT tout traitement IA, pour éviter
    de saturer la mémoire (cause principale des plantages 'out of memory')."""
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

    raw_bytes = await photo.read()  # gardé en mémoire uniquement

    try:
        original = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Image illisible ou corrompue.")

    # Downscale immédiat : on ne garde plus jamais les octets originaux
    # (potentiellement énormes) après cette étape.
    original = downscale_if_needed(original)
    del raw_bytes
    resized_buffer = io.BytesIO()
    original.save(resized_buffer, format="JPEG", quality=90)
    resized_bytes = resized_buffer.getvalue()
    resized_buffer.close()

    original_np = np.array(original)

    try:
        subject_rgba = remove_background(resized_bytes)
    except Exception as exc:
        logger.exception("Échec de la suppression d'arrière-plan")
        raise HTTPException(500, "Échec du traitement IA (arrière-plan).") from exc
    finally:
        del resized_bytes

    cropped = smart_crop_and_fit(subject_rgba, original_np)
    del subject_rgba, original_np, original
    cropped = apply_rounded_frame_mask(cropped)
    final_poster = compose_poster(cropped)
    del cropped

    # à ce stade, plus aucune grande image n'est retenue en mémoire — tout est
    # libéré immédiatement (RGPD + stabilité sur l'instance gratuite Render).
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
