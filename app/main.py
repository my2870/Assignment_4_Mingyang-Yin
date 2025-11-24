"""FastAPI application for generative models (GAN, EBM, Diffusion)"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import torch
import io
from PIL import Image
import os

from .helper_lib.model1 import get_model
from .helper_lib.generator import generate_samples, tensor_to_base64, tensor_to_pil

app = FastAPI(
    title="Generative Models API",
    description="API for generating images using GAN, Energy-Based Models, and Diffusion Models",
    version="1.0.0"
)

# Global variables for models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}


def load_models():
    """Load trained models"""
    global models
    
    # Support both local and Docker paths
    if os.path.exists("/app/models"):
        model_dir = "/app/models"
    elif os.path.exists("models"):
        model_dir = "models"
    else:
        # Try parent directory (for when running from app/)
        parent_models = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        if os.path.exists(parent_models):
            model_dir = parent_models
        else:
            print(f"Warning: models directory not found. Checked: /app/models, models, {parent_models}")
            print(f"Current directory: {os.getcwd()}")
            print(f"__file__: {__file__}")
            return
    
    # Load GAN
    if os.path.exists(f"{model_dir}/gan_generator.pth"):
        generator, _ = get_model('GAN', z_dim=100)
        generator.load_state_dict(torch.load(f"{model_dir}/gan_generator.pth", map_location=device))
        generator.to(device)
        generator.eval()
        models['GAN'] = generator
        print("✓ GAN model loaded")
    
    # Load EBM
    if os.path.exists(f"{model_dir}/ebm_model.pth"):
        ebm = get_model('EBM', num_channels=3)
        ebm.load_state_dict(torch.load(f"{model_dir}/ebm_model.pth", map_location=device))
        ebm.to(device)
        ebm.eval()
        models['EBM'] = ebm
        print("✓ EBM model loaded")
    
    # Load Diffusion
    if os.path.exists(f"{model_dir}/diffusion_model.pth"):
        diffusion = get_model('Diffusion', image_size=32, num_channels=3, embedding_dim=32)
        diffusion.load_state_dict(torch.load(f"{model_dir}/diffusion_model.pth", map_location=device))
        diffusion.to(device)
        diffusion.eval()
        models['Diffusion'] = diffusion
        print("✓ Diffusion model loaded")


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()
    print(f"Running on device: {device}")
    print(f"Available models: {list(models.keys())}")


class GenerateRequest(BaseModel):
    """Request model for image generation"""
    model_type: str = Field(..., description="Model type: 'GAN', 'EBM', or 'Diffusion'")
    num_samples: int = Field(default=4, ge=1, le=16, description="Number of images to generate")
    format: str = Field(default='base64', description="Output format: 'base64' or 'grid'")


class GenerateResponse(BaseModel):
    """Response model for image generation"""
    model_type: str
    num_samples: int
    images: list


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Generative Models API",
        "available_models": list(models.keys()),
        "endpoints": {
            "/generate": "Generate images from a model",
            "/models": "List available models",
            "/health": "Health check"
        }
    }


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": list(models.keys()),
        "device": str(device)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "device": str(device)
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_images(request: GenerateRequest):
    """
    Generate images using specified model
    
    - **model_type**: Type of model to use ('GAN', 'EBM', 'Diffusion')
    - **num_samples**: Number of images to generate (1-16)
    - **format**: Output format ('base64' or 'grid')
    """
    # Validate model type
    if request.model_type not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_type}' not available. Available models: {list(models.keys())}"
        )
    
    try:
        # Generate images
        model = models[request.model_type]
        images = generate_samples(
            model,
            model_type=request.model_type,
            num_samples=request.num_samples,
            device=device
        )
        
        # Convert to requested format
        if request.format == 'base64':
            image_data = tensor_to_base64(images)
        else:
            image_data = tensor_to_base64(images)
        
        return GenerateResponse(
            model_type=request.model_type,
            num_samples=request.num_samples,
            images=image_data
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating images: {str(e)}")


@app.get("/generate/{model_type}")
async def generate_images_get(
    model_type: str,
    num_samples: int = Query(default=4, ge=1, le=16)
):
    """
    Generate images using GET request (simpler interface)
    
    - **model_type**: Type of model to use ('GAN', 'EBM', 'Diffusion')
    - **num_samples**: Number of images to generate (1-16)
    """
    request = GenerateRequest(
        model_type=model_type,
        num_samples=num_samples,
        format='base64'
    )
    return await generate_images(request)


@app.get("/sample/{model_type}")
async def get_single_sample(model_type: str):
    """
    Get a single sample image as PNG
    
    - **model_type**: Type of model to use ('GAN', 'EBM', 'Diffusion')
    """
    if model_type not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_type}' not available. Available models: {list(models.keys())}"
        )
    
    try:
        # Generate single image
        model = models[model_type]
        images = generate_samples(
            model,
            model_type=model_type,
            num_samples=1,
            device=device
        )
        
        # Convert to PIL and return as PNG
        pil_images = tensor_to_pil(images)
        img_byte_arr = io.BytesIO()
        pil_images[0].save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

