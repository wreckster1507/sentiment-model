#!/usr/bin/env python3
"""
Production startup script for Sentiment Analysis API
Simplified startup with automatic environment setup
"""
import subprocess
import sys
import os

def start_production_server():
    """Start the production sentiment analysis server"""
    print("🚀 Starting Sentiment Analysis Production Server")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("local_deploy_v2.py"):
        print("❌ Error: local_deploy_v2.py not found")
        print("Make sure you're running this from the sentiment-model directory")
        return
    
    # Determine Python executable
    try:
        # For production deployment (like Render), use system Python
        if os.environ.get('PORT'):  # Render sets PORT environment variable
            python_cmd = 'python'
            print("🌐 Production environment detected (Render/Cloud)")
        elif os.name == 'nt':  # Windows local development
            if os.path.exists(r".\aisentiment\Scripts\python.exe"):
                python_cmd = r".\aisentiment\Scripts\python.exe"
                print("🖥️  Windows development environment")
                
                # Only set Windows-specific FFmpeg path for local development
                ffmpeg_path = os.path.expandvars(r"$LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1.1-full_build\bin")
                if os.path.exists(ffmpeg_path):
                    env = os.environ.copy()
                    env['PATH'] = ffmpeg_path + os.pathsep + env.get('PATH', '')
                    print(f"🎥 FFmpeg path set: {ffmpeg_path}")
                else:
                    env = None
                    print("⚠️  FFmpeg not found at expected Windows path, using system PATH")
            else:
                python_cmd = 'python'
                env = None
        else:
            # Linux/Mac - for production or local development
            if os.path.exists("./aisentiment/bin/python"):
                python_cmd = "./aisentiment/bin/python"
                print("🐧 Linux/Mac development environment")
            else:
                python_cmd = 'python'
                print("🐧 Linux/Mac production environment")
            env = None
        
        # Start server
        print(f"▶️  Starting server with: {python_cmd}")
        subprocess.run([python_cmd, "local_deploy_v2.py"], env=env)
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_production_server()
