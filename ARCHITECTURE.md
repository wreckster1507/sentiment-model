# 🏗️ Multimodal Sentiment Analysis - System Architecture

## 📋 Document Overview

This document provides a detailed technical architecture overview of the multimodal sentiment analysis system, including data flow, component interactions, and implementation details.

---

## 🎯 System Overview

### **High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                             │
│  Web Interface │ Mobile App │ API Clients │ Third Party     │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/REST API
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                  API GATEWAY                                │
│              FastAPI Server                                 │
│        • CORS handling                                      │
│        • Request validation                                 │
│        • File upload processing                             │
│        • Response formatting                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│               PROCESSING LAYER                              │
│                                                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │   Speech     │ │   Feature    │ │    Model            │ │
│  │ Transcription│ │  Extraction  │ │   Inference         │ │
│  │  (Whisper)   │ │  Pipeline    │ │  (PyTorch)          │ │
│  └──────────────┘ └──────────────┘ └──────────────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                   DATA LAYER                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │ Temporary    │ │   Model      │ │    Configuration     │ │
│  │ File Storage │ │  Weights     │ │       Data           │ │
│  │              │ │ (.pth files) │ │                      │ │
│  └──────────────┘ └──────────────┘ └──────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Architecture

### **1. Request Processing Flow**

```
📤 Client Upload
    │
    ▼
🔍 Input Validation
    │ • File type checking
    │ • Size validation  
    │ • Format verification
    ▼
💾 Temporary Storage
    │ • Save uploaded files
    │ • Generate unique IDs
    │ • Clean up policy
    ▼
🎵 Audio Extraction
    │ • FFmpeg processing
    │ • Format conversion
    │ • Quality optimization
    ▼
🗣️ Speech Transcription
    │ • Whisper model
    │ • Word-level timestamps
    │ • Language detection
    ▼
✂️ Segmentation
    │ • Split by utterances
    │ • Time boundaries
    │ • Text alignment
    ▼
🎯 Feature Extraction
    │ ├── Text: BERT tokenization
    │ ├── Video: Frame extraction
    │ └── Audio: Mel-spectrograms
    ▼
🧠 Model Inference
    │ • Multimodal fusion
    │ • Emotion prediction
    │ • Sentiment analysis
    ▼
📊 Response Generation
    │ • Confidence scoring
    │ • JSON formatting
    │ • Error handling
    ▼
📤 Client Response
```

---

## 🧠 Neural Network Architecture

### **Component Hierarchy**

```
MultimodalSentimentModel
├── TextEncoder (BERT-based)
│   ├── bert: AutoModel('bert-base-uncased')
│   │   ├── embeddings: BertEmbeddings
│   │   ├── encoder: BertEncoder (12 layers)
│   │   └── pooler: BertPooler
│   └── projection: Linear(768 → 128)
│
├── VideoEncoder (3D CNN)
│   └── r3d: R3D_18
│       ├── stem: Conv3d + BatchNorm3d + ReLU
│       ├── layer1-4: BasicBlock3d layers
│       ├── avgpool: AdaptiveAvgPool3d
│       └── fc: Linear(512 → 128)
│
├── AudioEncoder (1D CNN)
│   ├── conv_layers: Sequential
│   │   ├── Conv1d(64 → 64, kernel=3)
│   │   ├── BatchNorm1d(64)
│   │   ├── ReLU + MaxPool1d(2)
│   │   ├── Conv1d(64 → 128, kernel=3)
│   │   └── AdaptiveAvgPool1d(1)
│   └── projection: Linear(128 → 128)
│
├── fusion_layer: Sequential
│   ├── Linear(384 → 256)
│   ├── BatchNorm1d(256)
│   ├── ReLU()
│   ├── Dropout(0.3)
│   └── Linear(256 → 256)
│
├── emotion_classifier: Sequential
│   ├── Linear(256 → 128)
│   ├── ReLU()
│   ├── Dropout(0.2)
│   └── Linear(128 → 7)  # 7 emotion classes
│
└── sentiment_classifier: Sequential
    ├── Linear(256 → 128)
    ├── ReLU()
    ├── Dropout(0.2)
    └── Linear(128 → 3)  # 3 sentiment classes
```

### **Tensor Flow Dimensions**

```
Input Processing:
├── Text: [batch, seq_len] → tokenizer → [batch, 512] input_ids + attention_mask
├── Video: [batch, 30, 3, 224, 224] frames 
└── Audio: [batch, 64, 300] mel-spectrogram

Feature Extraction:
├── Text: [batch, 512] → BERT → [batch, 768] → projection → [batch, 128]
├── Video: [batch, 30, 3, 224, 224] → R3D → [batch, 512] → fc → [batch, 128]
└── Audio: [batch, 64, 300] → 1D CNN → [batch, 128] → projection → [batch, 128]

Fusion & Classification:
├── Concatenation: [batch, 128+128+128] = [batch, 384]
├── Fusion: [batch, 384] → fusion_layer → [batch, 256]
├── Emotions: [batch, 256] → emotion_classifier → [batch, 7]
└── Sentiments: [batch, 256] → sentiment_classifier → [batch, 3]

Output Processing:
├── Softmax: logits → probabilities
├── ArgMax: probabilities → class indices
└── Mapping: indices → label strings
```

---

## ⚙️ Component Details

### **1. Speech Processing Pipeline**

```python
# Architecture Flow
Video File
    │ FFmpeg
    ▼
Audio Stream (16kHz, mono)
    │ Whisper
    ▼
Transcription + Timestamps
    │ Segmentation Algorithm
    ▼
Utterances with Time Boundaries
    │ For each utterance:
    ├── Extract corresponding video frame
    ├── Extract audio segment  
    ├── Process text through BERT
    └── Combine for multimodal analysis
```

**Key Components:**
- **Whisper Model**: `whisper.load_model("base")` for speech-to-text
- **FFmpeg Integration**: Audio extraction and format conversion
- **Timestamp Alignment**: Word-level precision for segmentation
- **Utterance Detection**: Natural pause-based splitting

### **2. Multimodal Feature Extraction**

#### **Text Processing**
```python
# BERT Tokenization and Encoding
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Process text
inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
text_features = outputs.pooler_output  # [CLS] token representation
```

#### **Video Processing**  
```python
# Frame Extraction and 3D CNN
cap = cv2.VideoCapture(video_path)
frames = []
for i in range(30):  # Extract 30 frames
    ret, frame = cap.read()
    frame = cv2.resize(frame, (224, 224))
    frames.append(frame)

video_tensor = torch.FloatTensor(frames).permute(0, 3, 1, 2)  # [30, 3, 224, 224]
video_features = r3d_model(video_tensor.unsqueeze(0))  # Add batch dimension
```

#### **Audio Processing**
```python
# Mel-Spectrogram Generation
audio, sr = librosa.load(audio_path, sr=16000)
mel_spec = librosa.feature.melspectrogram(
    y=audio, 
    sr=sr, 
    n_mels=64, 
    n_fft=2048, 
    hop_length=512
)
audio_features = torch.FloatTensor(mel_spec).unsqueeze(0)  # Add batch dimension
```

---

## 🔧 System Configuration

### **Environment Variables**
```python
# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()

# Model Paths
MODEL_PATH = "path/to/model.pth"
WHISPER_MODEL = "base"  # or "small", "medium", "large"

# Processing Parameters
MAX_FRAMES = 30
VIDEO_SIZE = (224, 224)
AUDIO_SAMPLE_RATE = 16000
MEL_BINS = 64
TEXT_MAX_LENGTH = 512

# API Configuration
API_HOST = "127.0.0.1"
API_PORT = 8000
CORS_ORIGINS = ["http://localhost:3000"]
```

### **Memory Management**
```python
# Automatic cleanup of temporary files
@contextmanager
def temp_file_manager(suffix='.mp4'):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        yield temp_file.name
    finally:
        os.unlink(temp_file.name)

# GPU memory optimization
torch.cuda.empty_cache()  # Clear GPU cache
```

---

## 📊 Performance Characteristics

### **Throughput Metrics**
- **CPU Processing**: 10-30 seconds per video (2-3 minutes)
- **GPU Processing**: 3-10 seconds per video (2-3 minutes)  
- **Concurrent Requests**: FastAPI supports async processing
- **Memory Usage**: 2-4GB RAM during processing

### **Scalability Considerations**
- **Horizontal Scaling**: Multiple server instances behind load balancer
- **Vertical Scaling**: GPU acceleration, more RAM
- **Caching Strategy**: Redis for frequent requests
- **Database Integration**: Store results for repeat analysis

### **Quality Metrics**
- **Speech Recognition**: Depends on audio quality and language
- **Emotion Detection**: Model accuracy varies by domain
- **Sentiment Analysis**: Generally high accuracy for clear text
- **Confidence Scores**: Provided for each prediction

---

## 🛡️ Error Handling & Resilience

### **Error Categories**

1. **Input Validation Errors**
   - Invalid file formats
   - File size limitations
   - Corrupted uploads

2. **Processing Errors**
   - FFmpeg failures
   - Whisper transcription errors
   - Model inference failures

3. **System Errors**
   - Out of memory
   - Disk space issues
   - Network timeouts

### **Recovery Strategies**

```python
# Graceful degradation
try:
    utterances = process_video_segments(video_path)
except Exception as e:
    logger.warning(f"Segmentation failed: {e}")
    # Fallback to single-segment analysis
    utterances = create_fallback_utterance(video_path)

# Retry mechanisms
@retry(max_attempts=3, backoff=2.0)
def transcribe_audio(audio_path):
    return whisper_model.transcribe(audio_path)
```

---

## 🔄 Future Architecture Enhancements

### **Planned Improvements**
1. **Real-time Processing**: WebSocket support for live video streams
2. **Model Optimization**: Quantization and pruning for faster inference
3. **Multi-language Support**: Extended Whisper model integration
4. **Cloud Integration**: AWS/GCP deployment with auto-scaling
5. **Advanced Caching**: Intelligent result caching and invalidation

### **Extension Points**
- **Plugin Architecture**: Support for custom emotion models
- **API Versioning**: Backward compatibility for API changes
- **Monitoring Integration**: Prometheus/Grafana metrics
- **Security Layer**: Authentication and rate limiting

---

## 📝 Configuration Files

### **requirements.txt Structure**
```
# Core ML Framework
torch>=1.13.0
torchvision>=0.14.0
torchaudio>=0.13.0

# Transformers & NLP
transformers>=4.21.0
openai-whisper>=20230314

# Web Framework
fastapi>=0.95.0
uvicorn>=0.20.0

# Media Processing
opencv-python>=4.7.0
librosa>=0.9.0
ffmpeg-python>=0.2.0

# Utilities
numpy>=1.21.0
requests>=2.28.0
```

This architecture provides a robust, scalable foundation for multimodal sentiment analysis with clear separation of concerns and extensibility for future enhancements.
