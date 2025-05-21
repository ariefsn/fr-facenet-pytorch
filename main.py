import base64
import io
import time
from typing import Any, Dict, Optional

import numpy as np
import requests
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from torchvision import transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("DEVICE", DEVICE)

app = FastAPI(title="Face Recognition API with FaceNet512")

# Initialize MTCNN for face detection and landmarks, InceptionResnetV1 for embeddings
mtcnn = MTCNN(keep_all=False, device=DEVICE)  # change to 'cuda' if GPU available
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

THRESHOLD = 0.4  # distance threshold for verification (cosine distance)

# Pydantic models
class ImageInput(BaseModel):
    imageSource: str = Field(..., description="Base64 string or URL of source image")
    imageTarget: str = Field(..., description="Base64 string or URL of target image")

class FacialArea(BaseModel):
    x: int
    y: int
    w: int
    h: int
    leftEye: Optional[Dict[str, int]] = None
    rightEye: Optional[Dict[str, int]] = None

class FacialAreas(BaseModel):
    imgSource: Optional[FacialArea]
    imgTarget: Optional[FacialArea]

class VerifyResponse(BaseModel):
    verified: bool
    facialAreas: FacialAreas
    distance: float
    threshold: float
    model: str
    detectorBackend: str
    similarityMetric: str
    time: float


def load_image(image_str: str) -> Image.Image:
    """Load an image from base64 or URL"""
    if image_str.startswith('http'):
        # Load from URL
        resp = requests.get(image_str)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to load image from URL")
        return Image.open(io.BytesIO(resp.content)).convert('RGB')
    else:
        # Assume base64
        try:
            header, encoded = image_str.split(',', 1) if ',' in image_str else (None, image_str)
            img_bytes = base64.b64decode(encoded)
            return Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to decode base64 image")


def get_face_data(image: Image.Image) -> (np.ndarray, Optional[Dict[str, Any]]):
    """Detect face, landmarks and extract embedding"""
    # Detect face + landmarks
    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)

    if boxes is None or len(boxes) == 0:
        return None, None

    box = boxes[0]  # single face only (first detected)
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1

    facial_area = {
        "x": x1,
        "y": y1,
        "w": w,
        "h": h,
        "leftEye": None,
        "rightEye": None,
    }
    if landmarks is not None and len(landmarks) > 0:
        # landmarks[0] corresponds to first face
        lmk = landmarks[0]
        # landmarks order: left_eye, right_eye, nose, mouth_left, mouth_right
        facial_area["leftEye"] = {"x": int(lmk[0][0]), "y": int(lmk[0][1])}
        facial_area["rightEye"] = {"x": int(lmk[1][0]), "y": int(lmk[1][1])}

    # Crop and preprocess face for embedding
    face_img = image.crop((x1, y1, x2, y2)).resize((160, 160))  # 160x160 for FaceNet input
    # face_tensor = mtcnn.transforms(face_img).unsqueeze(0)  # shape: (1, 3, 160, 160)
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    face_tensor = preprocess(face_img).unsqueeze(0)

    # Get embedding
    embedding = resnet(face_tensor).detach().cpu().numpy()[0]

    return embedding, facial_area


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return 1 - np.dot(a_norm, b_norm)


@app.post("/verify", response_model=VerifyResponse, summary="Verify face similarity")
def verify_faces(payload: ImageInput):
    start_time = time.time()

    img_src = load_image(payload.imageSource)
    img_tgt = load_image(payload.imageTarget)

    emb_src, area_src = get_face_data(img_src)
    emb_tgt, area_tgt = get_face_data(img_tgt)

    if emb_src is None or emb_tgt is None:
        raise HTTPException(status_code=400, detail="No face detected in one or both images")

    dist = cosine_distance(emb_src, emb_tgt)
    verified = dist < THRESHOLD

    elapsed = time.time() - start_time

    return VerifyResponse(
        verified=verified,
        facialAreas=FacialAreas(imgSource=area_src, imgTarget=area_tgt),
        distance=float(dist),
        threshold=THRESHOLD,
        model="FaceNet512",
        detectorBackend="MTCNN",
        similarityMetric="cosine",
        time=round(elapsed, 4)
    )
