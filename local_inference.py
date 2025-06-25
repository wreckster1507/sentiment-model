"""
Local inference utilities - S3-free version
This file contains only the components needed for local deployment
"""
import torch
import cv2
import numpy as np
import subprocess
import torchaudio
import os
import shutil
from transformers import AutoTokenizer


EMOTION_MAP = {0: "anger", 1: "disgust", 2: "fear",
               3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def get_ffmpeg_path():
    """Get FFmpeg executable path, checking multiple common locations"""
    # First, check if ffmpeg is in PATH
    if shutil.which('ffmpeg'):
        return 'ffmpeg'
    
    # Common Windows locations
    common_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        os.path.expandvars(r"$LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe")
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    # For production deployment (like Render), ffmpeg should be in PATH
    return 'ffmpeg'


class VideoProcessor:
    """Process video files locally - no S3 dependency"""
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")

            # Try and read first frame to validate video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")

            # Reset index to not skip first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            cap.release()

        if (len(frames) == 0):
            raise ValueError("No frames could be extracted")

        # Pad or truncate frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        # Before permute: [frames, height, width, channels]
        # After permute: [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)


class AudioProcessor:
    """Process audio files locally - no S3 dependency"""
    def extract_features(self, audio_path, max_length=300):
        try:
            # Load audio directly if it's already a .wav file
            if audio_path.endswith('.wav'):
                waveform, sample_rate = torchaudio.load(audio_path)
            else:
                # Convert to wav first if it's a different format
                temp_wav = audio_path.replace(os.path.splitext(audio_path)[1], '_temp.wav')
                ffmpeg_cmd = get_ffmpeg_path()
                subprocess.run([
                    ffmpeg_cmd,
                    '-i', audio_path,
                    '-vn',
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    temp_wav
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                waveform, sample_rate = torchaudio.load(temp_wav)
                os.remove(temp_wav)  # Clean up temp file

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )

            mel_spec = mel_spectrogram(waveform)

            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < max_length:
                padding = max_length - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :max_length]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")


def process_text_local(text: str):
    """Process text locally without any S3 dependency"""
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    text_inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return text_inputs


# Example of how to use these processors locally
if __name__ == "__main__":
    print("Local processors initialized successfully!")
    print("No S3 or cloud dependencies required.")
    
    # Example usage:
    video_processor = VideoProcessor()
    audio_processor = AudioProcessor()
    
    print("VideoProcessor: Ready for local video files")
    print("AudioProcessor: Ready for local audio files")
    print("TextProcessor: Ready for text input")
