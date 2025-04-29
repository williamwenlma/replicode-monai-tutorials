import os
import torch
import nibabel as nib
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    NormalizeIntensity,
    Orientation,
    Spacing,
    EnsureType,
)
from monai.inferers import sliding_window_inference
from typing import List
import tempfile

app = FastAPI(title="BraTS Segmentation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize device and model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
).to(device)

# Load model weights
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.pt")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define inference transforms
inference_transform = Compose([
    EnsureChannelFirst(),
    EnsureType(),
    Orientation(axcodes="RAS"),
    Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    NormalizeIntensity(nonzero=True, channel_wise=True),
])

# 添加根路径接口
@app.get("/")
def root():
    return {"status": "ok", "message": "BraTS Segmentation API is running"}

# Add an endpoint to check model status
@app.get("/model-status")
def model_status():
    try:
        return {
            "status": "ok",
            "device": str(device),
            "model_path": str(model_path),
            "model_loaded": model is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    """
    Endpoint for BraTS segmentation prediction.
    Expects 4 NIfTI files (t1, t1ce, t2, flair)
    Returns segmentation masks for TC, WT, and ET
    """

    if len(files) != 4:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 4 files (t1, t1ce, t2, flair), but got {len(files)}"
        )

    try:
        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files
            file_paths = []
            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                file_paths.append(file_path)

            # Load and stack images
            images = []
            for path in file_paths:
                img = LoadImage()(path)
                images.append(img[0])
            
            # Stack images and apply transforms
            input_data = np.stack(images)
            input_tensor = inference_transform(input_data)
            input_tensor = input_tensor.unsqueeze(0).to(device)

            # Run inference
            with torch.no_grad():
                roi_size = (128, 128, 64)
                sw_batch_size = 4
                output = sliding_window_inference(
                    input_tensor, roi_size, sw_batch_size, model
                )
                output = torch.sigmoid(output)

            # Convert output to numpy
            output_np = output.cpu().numpy()

            # Create result dictionary
            result = {
                "tumor_core": output_np[0, 0].tolist(),
                "whole_tumor": output_np[0, 1].tolist(),
                "enhancing_tumor": output_np[0, 2].tolist()
            }

            return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015)