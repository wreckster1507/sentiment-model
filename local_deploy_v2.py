"""
Updated local deployment with time-segmented speech analysis
This version provides multiple utterances with speech transcription to match the frontend UI
"""
import os
import torch
import tempfile
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from models import MultimodalSentimentModel
import json
from local_inference import VideoProcessor, AudioProcessor, process_text_local
from transformers import AutoTokenizer
from typing import Optional
import logging
import numpy as np
import cv2
import whisper

# GPU Configuration - Prioritize GPU, fallback to CPU
def setup_device():
    """Setup device prioritizing GPU over CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        return device
    else:
        device = torch.device("cpu")
        print("⚠️  GPU not available, using CPU as fallback")
        return device

# Global device
DEVICE = setup_device()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU Configuration
def setup_device():
    """Setup device (GPU if available, CPU as fallback)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🚀 GPU DETECTED: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        logger.info(f"🔥 CUDA Version: {torch.version.cuda}")
        logger.info("✅ Using GPU acceleration for faster processing!")
        return device, True
    else:
        device = torch.device("cpu")
        logger.info("⚠️  GPU not available, using CPU (slower)")
        logger.info("💡 Install CUDA PyTorch for GPU acceleration: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        return device, False

# Initialize device
DEVICE, USE_GPU = setup_device()

# Global variables for model and processors
model = None
video_processor = None
audio_processor = None
tokenizer = None
whisper_model = None

EMOTION_MAP = {0: "anger", 1: "disgust", 2: "fear", 
               3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}

def get_text_features(text: str):
    """Process text and return tokenized features"""
    return process_text_local(text)

def get_audio_features(audio_path: str):
    """Process audio file and return features"""
    global audio_processor
    if audio_processor is None:
        return np.zeros((1, 64, 100))  # Return dummy features if processor not available
    
    try:
        features = audio_processor.extract_features(audio_path)
        return features.numpy()
    except Exception as e:
        logger.warning(f"Audio processing failed: {str(e)}, using dummy features")
        return np.zeros((1, 64, 100))

def get_video_duration(video_path: str) -> float:
    """Get the duration of a video file in seconds"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 30.0  # Default fallback
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if fps > 0:
            duration = frame_count / fps
            return float(duration)
        else:
            return 30.0  # Default fallback
    except Exception:
        return 30.0  # Default fallback

def process_video_segments(video_path: str):
    """Process video into time segments with speech transcription and emotion/sentiment analysis"""
    global whisper_model, model
    
    try:
        # Transcribe with word-level timestamps
        logger.info("Transcribing video with Whisper...")
        
        # Set FFmpeg path for Whisper
        import os
        original_path = os.environ.get('PATH', '')
        
        # Add potential FFmpeg paths to environment
        if os.name == 'nt':  # Windows
            potential_ffmpeg_paths = [
                os.path.expandvars(r"$LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1.1-full_build\bin"),
                r"C:\ffmpeg\bin",
                r"C:\Program Files\ffmpeg\bin"
            ]
            
            for path in potential_ffmpeg_paths:
                if os.path.exists(path):
                    os.environ['PATH'] = path + os.pathsep + original_path
                    logger.info(f"Added FFmpeg path: {path}")
                    break
        
        try:
            result = whisper_model.transcribe(video_path, word_timestamps=True, verbose=False)
        finally:
            # Restore original PATH
            os.environ['PATH'] = original_path
        
        logger.info(f"Whisper transcription completed. Found {len(result.get('segments', []))} segments")
        
        utterances = []
        
        # Check if we got any segments
        if not result.get("segments"):
            logger.warning("No speech segments detected by Whisper")
            return []
        
        # Process each segment
        for i, segment in enumerate(result["segments"]):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", start_time + 5)  # Default 5 second duration
            text = segment.get("text", "").strip()
            
            logger.info(f"Processing segment {i+1}/{len(result['segments'])}: {start_time:.1f}-{end_time:.1f}s: '{text[:50]}...'")
            
            if len(text) < 2:  # Skip very short segments (was 3, now 2 to catch more)
                logger.info(f"Skipping short segment: '{text}'")
                continue
                
            try:
                # Extract video segment (simplified - use middle frame)
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                  # Get frame at middle of segment
                middle_time = (start_time + end_time) / 2
                frame_number = int(middle_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    # Process frame for video features
                    frame = cv2.resize(frame, (224, 224))
                    frame = frame / 255.0
                    # Create 30 frames by repeating this frame
                    frames = [frame] * 30
                    video_tensor = torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
                    video_features = video_tensor.unsqueeze(0).to(DEVICE)  # Move to GPU/CPU
                else:
                    # Use dummy video features
                    video_features = torch.zeros((1, 30, 3, 224, 224), dtype=torch.float32).to(DEVICE)
                
                # Process text
                text_features = process_text_local(text)
                # Move text features to device
                if isinstance(text_features, dict):
                    text_features = {k: v.to(DEVICE) for k, v in text_features.items()}
                else:
                    text_features = text_features.to(DEVICE)
                  # Use dummy audio features for now (can be enhanced)
                audio_features = torch.zeros((1, 64, 100), dtype=torch.float32).to(DEVICE)
                
                # Run model inference
                with torch.no_grad():
                    outputs = model(text_features, video_features, audio_features)
                    
                    emotion_probs = torch.softmax(outputs['emotions'], dim=1)
                    sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)
                    
                    # Convert to sorted arrays
                    emotions_array = []
                    for i in range(len(EMOTION_MAP)):
                        emotions_array.append({
                            "label": EMOTION_MAP[i],
                            "confidence": float(emotion_probs[0][i])
                        })
                    emotions_array.sort(key=lambda x: x["confidence"], reverse=True)
                    
                    sentiments_array = []
                    for i in range(len(SENTIMENT_MAP)):
                        sentiments_array.append({
                            "label": SENTIMENT_MAP[i],
                            "confidence": float(sentiment_probs[0][i])
                        })
                    sentiments_array.sort(key=lambda x: x["confidence"], reverse=True)
                    
                    # Add utterance
                    utterances.append({
                        "start_time": round(start_time, 1),
                        "end_time": round(end_time, 1),
                        "text": text,
                        "emotions": emotions_array,
                        "sentiments": sentiments_array                    })
                    
                    logger.info(f"Successfully processed segment {start_time:.1f}-{end_time:.1f}: {text[:50]}...")
                    
            except Exception as e:
                logger.warning(f"Failed to process segment {start_time}-{end_time}: {str(e)}")
                continue
        
        logger.info(f"Successfully processed {len(utterances)} utterances")
        
        # Debug logging
        logger.info(f"DEBUG: About to return {len(utterances)} utterances")
        if utterances:
            logger.info(f"DEBUG: First utterance: {utterances[0]['text'][:50]}...")
            logger.info(f"DEBUG: First utterance emotions: {len(utterances[0].get('emotions', []))}")
            logger.info(f"DEBUG: First utterance sentiments: {len(utterances[0].get('sentiments', []))}")
        
        return utterances
        
    except Exception as e:
        logger.error(f"Video segmentation failed: {str(e)}")
        return []

def load_model(model_path: str):
    """Load the trained model from the specified path"""
    global model
    
    try:
        # Load model
        model = MultimodalSentimentModel()
        
        # Load the state dict
        if os.path.exists(model_path):
            if USE_GPU:
                logger.info("🚀 Loading model weights on GPU...")
                state_dict = torch.load(model_path, map_location=DEVICE)
            else:
                logger.info("💻 Loading model weights on CPU...")
                state_dict = torch.load(model_path, map_location='cpu')  # CPU fallback
                
            model.load_state_dict(state_dict)
            
            # Move model to device
            model = model.to(DEVICE)
            model.eval()
            
            if USE_GPU:
                logger.info(f"✅ Model loaded successfully on GPU from {model_path}")
            else:
                logger.info(f"✅ Model loaded successfully on CPU from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def initialize_processors():
    """Initialize video, audio processors and whisper model"""
    global video_processor, audio_processor, whisper_model
    
    try:
        video_processor = VideoProcessor()
        audio_processor = AudioProcessor()
        
        # Load Whisper model for speech transcription
        logger.info("Loading Whisper model...")
        if USE_GPU:
            logger.info("🚀 Loading Whisper on GPU for faster transcription...")
            whisper_model = whisper.load_model("base", device=DEVICE)
        else:
            logger.info("💻 Loading Whisper on CPU...")
            whisper_model = whisper.load_model("base")  # CPU fallback
            
        logger.info("Processors and Whisper model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing processors: {str(e)}")
        return False

# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Local deployment of multimodal sentiment analysis model with speech segmentation",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.on_event("startup")
async def startup_event():
    """Initialize model and processors on startup"""
    model_path = "model/model.pth"  # Use relative path for production deployment
    
    if not load_model(model_path):
        logger.error("Failed to load model on startup")
        raise RuntimeError("Model loading failed")
    
    if not initialize_processors():
        logger.error("Failed to initialize processors on startup")
        raise RuntimeError("Processor initialization failed")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Sentiment Analysis API with Speech Segmentation is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "processors_loaded": video_processor is not None and audio_processor is not None,
        "whisper_loaded": whisper_model is not None
    }

@app.post("/predict")
async def predict_sentiment(
    video_file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    """
    Predict sentiment and emotion from multimodal input with time segmentation
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not any([video_file, audio_file, text]):
        raise HTTPException(
            status_code=400, 
            detail="At least one input (video, audio, or text) is required"
        )
    
    try:
        # Process inputs
        text_features = None
        utterances = []
        
        # Process text if provided
        if text:
            text_features = get_text_features(text)
        else:
            # Create dummy text features if not provided
            text_features = {
                'input_ids': torch.zeros((1, 512), dtype=torch.long),
                'attention_mask': torch.zeros((1, 512), dtype=torch.long)
            }
          # Process video if provided
        if video_file:
            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                shutil.copyfileobj(video_file.file, tmp_video)
                tmp_video_path = tmp_video.name
            
            try:
                # Process video into time segments with speech transcription
                logger.info("Processing video segments with speech transcription...")
                utterances = process_video_segments(tmp_video_path)
                
                if not utterances:
                    # Better fallback: Create time-based segments even without speech transcription
                    logger.warning("Video segmentation failed or no speech detected, creating time-based segments")
                    estimated_duration = get_video_duration(tmp_video_path)
                    
                    # Create 30-second segments for analysis
                    segment_duration = 30.0
                    num_segments = max(1, int(estimated_duration / segment_duration))
                    
                    for i in range(num_segments):
                        segment_start = i * segment_duration
                        segment_end = min((i + 1) * segment_duration, estimated_duration)
                        
                        logger.info(f"Creating fallback segment {i+1}/{num_segments}: {segment_start:.1f}-{segment_end:.1f}s")
                        
                        # Extract video features for this segment
                        cap = cv2.VideoCapture(tmp_video_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        # Get frame from middle of segment
                        middle_time = (segment_start + segment_end) / 2
                        frame_number = int(middle_time * fps)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret:
                            # Process frame for video features
                            frame = cv2.resize(frame, (224, 224))
                            frame = frame / 255.0
                            # Create 30 frames by repeating this frame
                            frames = [frame] * 30
                            video_tensor = torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
                            video_features = video_tensor.unsqueeze(0).to(DEVICE)
                        else:
                            # Use dummy video features
                            video_features = torch.zeros((1, 30, 3, 224, 224), dtype=torch.float32).to(DEVICE)
                        
                        audio_features = torch.zeros((1, 64, 100), dtype=torch.float32).to(DEVICE)
                        
                        # Use dummy text for this segment
                        dummy_text = f"Video segment {i+1} analysis"
                        segment_text_features = process_text_local(dummy_text)
                        
                        # Move text features to device
                        if isinstance(segment_text_features, dict):
                            segment_text_features = {k: v.to(DEVICE) for k, v in segment_text_features.items()}
                        else:
                            segment_text_features = segment_text_features.to(DEVICE)
                        
                        # Create segment analysis
                        with torch.no_grad():
                            outputs = model(segment_text_features, video_features, audio_features)
                            emotion_probs = torch.softmax(outputs['emotions'], dim=1)
                            sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)
                            
                            emotions_array = []
                            for j in range(len(EMOTION_MAP)):
                                emotions_array.append({
                                    "label": EMOTION_MAP[j],
                                    "confidence": float(emotion_probs[0][j])
                                })
                            emotions_array.sort(key=lambda x: x["confidence"], reverse=True)
                            
                            sentiments_array = []
                            for j in range(len(SENTIMENT_MAP)):
                                sentiments_array.append({
                                    "label": SENTIMENT_MAP[j],
                                    "confidence": float(sentiment_probs[0][j])
                                })
                            sentiments_array.sort(key=lambda x: x["confidence"], reverse=True)
                            
                            utterances.append({
                                "start_time": round(segment_start, 1),
                                "end_time": round(segment_end, 1),
                                "text": f"Segment {i+1}: Visual analysis - {emotions_array[0]['label']} emotion, {sentiments_array[0]['label']} sentiment",
                                "emotions": emotions_array,
                                "sentiments": sentiments_array
                            })
                    
                    logger.info(f"Created {len(utterances)} time-based fallback segments")                
            finally:
                os.unlink(tmp_video_path)  # Clean up temp file
        else:
            # Text-only or audio-only processing
            video_features = torch.zeros((1, 30, 3, 224, 224), dtype=torch.float32).to(DEVICE)
            audio_features = torch.zeros((1, 1, 64, 100), dtype=torch.float32).to(DEVICE)
            
            # Move text features to device
            if isinstance(text_features, dict):
                text_features = {k: v.to(DEVICE) for k, v in text_features.items()}
            else:
                text_features = text_features.to(DEVICE)
            
            # Single utterance for text-only
            with torch.no_grad():
                outputs = model(text_features, video_features, audio_features)
                emotion_probs = torch.softmax(outputs['emotions'], dim=1)
                sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)
                
                emotions_array = []
                for i in range(len(EMOTION_MAP)):
                    emotions_array.append({
                        "label": EMOTION_MAP[i],
                        "confidence": float(emotion_probs[0][i])
                    })
                emotions_array.sort(key=lambda x: x["confidence"], reverse=True)
                
                sentiments_array = []
                for i in range(len(SENTIMENT_MAP)):
                    sentiments_array.append({
                        "label": SENTIMENT_MAP[i],
                        "confidence": float(sentiment_probs[0][i])
                    })
                sentiments_array.sort(key=lambda x: x["confidence"], reverse=True)
                
                utterances = [{
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "text": text or "Text analysis results",
                    "emotions": emotions_array,
                    "sentiments": sentiments_array
                }]
          # Return response in required format
        response = {
            "analysis": {
                "utterances": utterances
            }
        }
        
        # Debug the final response
        logger.info(f"DEBUG: Final response contains {len(utterances)} utterances")
        if utterances:
            logger.info(f"DEBUG: Final first utterance text: {utterances[0]['text'][:50]}...")
        
        return JSONResponse(content=response)
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_text")
async def predict_text_only(text: str = Form(...)):
    """
    Predict sentiment and emotion from text only
    """
    return await predict_sentiment(text=text)

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # Use environment variables for production deployment (Render)
    import os
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    run_server(host=host, port=port)
