import os
import shutil
import sys
import zipfile
import argparse
import subprocess
import tempfile
import requests
import gradio as gr

from rvc.modules.download_source import download_file

# Path to the directory where RVC models will be stored
rvc_models_dir = os.path.join(os.getcwd(), "models", "RVC_models")
os.makedirs(rvc_models_dir, exist_ok=True)

# Enhanced model manager with PolUVR integration
class EnhancedModelManager:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.poluvr_available = self._check_poluvr_availability()
        
    def _check_poluvr_availability(self):
        """Check if PolUVR is available for processing"""
        try:
            import PolUVR
            return True
        except ImportError:
            return False
    
    def apply_poluvr_processing(self, audio_path, model_name="UVR-MDX-NET 1 2 3"):
        """Apply PolUVR source separation to audio file"""
        if not self.poluvr_available:
            print("PolUVR not available, skipping preprocessing")
            return audio_path
            
        try:
            from PolUVR import PolUVR
            print(f"🎛️  Applying PolUVR processing with model: {model_name}")
            
            # Initialize PolUVR
            uvr = PolUVR()
            
            # Create output path
            processed_path = os.path.join(
                self.temp_dir, 
                f"poluvr_{os.path.basename(audio_path)}"
            )
            
            # Process audio
            uvr.separate(
                input_path=audio_path,
                output_path=processed_path,
                model_name=model_name,
                gpu=True  # Use GPU if available
            )
            
            print(f"✅ PolUVR processing completed: {processed_path}")
            return processed_path
            
        except Exception as e:
            print(f"⚠️  PolUVR processing failed: {e}")
            print("Continuing with original audio...")
            return audio_path

# Initialize enhanced model manager
enhanced_manager = EnhancedModelManager()

# Enhanced extraction function with PolUVR support
def extract_zip(extraction_folder, zip_name, apply_poluvr=False, poluvr_model="UVR-MDX-NET 1 2 3"):
    os.makedirs(extraction_folder, exist_ok=True)  # Create the extraction directory if it doesn't exist
    
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall(extraction_folder)  # Extract the zip file
    os.remove(zip_name)  # Delete the zip file after extraction

    index_filepath, model_filepath = None, None
    audio_files = []
    
    # Iterate through all files in the extracted directory to find .pth, .index, and audio files
    for root, _, files in os.walk(extraction_folder):
        for name in files:
            file_path = os.path.join(root, name)
            file_size = os.stat(file_path).st_size
            
            # Check for model files
            if name.endswith(".index") and file_size > 1024 * 100:  # Minimum size for index file
                index_filepath = file_path
            if name.endswith(".pth") and file_size > 1024 * 1024 * 40:  # Minimum size for pth file
                model_filepath = file_path
            
            # Check for audio files for PolUVR processing
            if name.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')) and file_size > 1024 * 1024:  # Audio files > 1MB
                audio_files.append(file_path)

    if not model_filepath:
        # If no model file is found, raise an error
        raise gr.Error(f"No .pth model file found in the extracted zip. Check the contents in {extraction_folder}.")

    # Apply PolUVR processing if requested and audio files are found
    if apply_poluvr and audio_files:
        print(f"🎛️  Found {len(audio_files)} audio files for PolUVR processing")
        for audio_file in audio_files:
            try:
                processed_file = enhanced_manager.apply_poluvr_processing(audio_file, poluvr_model)
                if processed_file != audio_file:
                    # Replace original with processed version
                    shutil.move(processed_file, audio_file)
                    print(f"✅ Processed: {os.path.basename(audio_file)}")
            except Exception as e:
                print(f"⚠️  Failed to process {os.path.basename(audio_file)}: {e}")
                continue

    # Rename files and clean up unnecessary folders
    rename_and_cleanup(extraction_folder, model_filepath, index_filepath)


# Function to rename files and remove empty folders
def rename_and_cleanup(extraction_folder, model_filepath, index_filepath):
    os.rename(
        model_filepath,
        os.path.join(extraction_folder, os.path.basename(model_filepath)),
    )
    if index_filepath:
        os.rename(
            index_filepath,
            os.path.join(extraction_folder, os.path.basename(index_filepath)),
        )

    # Remove remaining empty directories after extraction
    for filepath in os.listdir(extraction_folder):
        full_path = os.path.join(extraction_folder, filepath)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)


# Enhanced download function with PolUVR support
def enhanced_download_from_url(url, dir_name, progress=gr.Progress(), use_poluvr=False, poluvr_model="UVR-MDX-NET 1 2 3"):
    try:
        progress(0, desc=f"[~] Downloading voice model {dir_name}...")
        zip_name = os.path.join(rvc_models_dir, dir_name + ".zip")
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        
        if os.path.exists(extraction_folder):
            # Check if a directory with the same name already exists
            raise gr.Error(f"Voice model directory {dir_name} already exists! Choose a different name for your voice model.")

        download_file(url, zip_name, progress)  # Download the file
        progress(0.8, desc="[~] Extracting zip file...")
        
        # Extract with PolUVR processing if requested
        extract_zip(extraction_folder, zip_name, use_poluvr, poluvr_model)
        
        if use_poluvr:
            progress(0.9, desc="[~] PolUVR processing completed")
        
        return f"[+] Model {dir_name} successfully downloaded!"
    except Exception as e:
        # Handle errors during model download
        raise gr.Error(f"Error downloading model: {str(e)}")


# Function to upload and extract a model zip file through the interface
def enhanced_upload_zip_file(zip_path, dir_name, progress=gr.Progress(), use_poluvr=False, poluvr_model="UVR-MDX-NET 1 2 3"):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f"Voice model directory {dir_name} already exists! Choose a different name for your voice model.")

        zip_name = zip_path.name
        progress(0.8, desc="[~] Extracting zip file...")
        extract_zip(extraction_folder, zip_name, use_poluvr, poluvr_model)  # Extract the zip file
        return f"[+] Model {dir_name} successfully uploaded!"
    except Exception as e:
        # Handle errors during upload and extraction
        raise gr.Error(f"Error uploading model: {str(e)}")


# Function to upload separate model files (.pth and .index)
def enhanced_upload_separate_files(pth_file, index_file, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f"Voice model directory {dir_name} already exists! Choose a different name for your voice model.")

        os.makedirs(extraction_folder, exist_ok=True)

        # Upload .pth file
        progress(0.4, desc="[~] Uploading .pth file...")
        if pth_file:
            pth_path = os.path.join(extraction_folder, os.path.basename(pth_file.name))
            shutil.copyfile(pth_file.name, pth_path)

        # Upload .index file
        progress(0.8, desc="[~] Uploading .index file...")
        if index_file:
            index_path = os.path.join(extraction_folder, os.path.basename(index_file.name))
            shutil.copyfile(index_file.name, index_path)

        return f"[+] Model {dir_name} successfully uploaded!"
    except Exception as e:
        # Handle errors during file upload
        raise gr.Error(f"Error uploading model: {str(e)}")


# Enhanced command-line interface
def create_parser():
    """Create enhanced argument parser for command-line interface"""
    parser = argparse.ArgumentParser(
        description="Enhanced RVC Model Manager with PolUVR integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a model from URL
  python3 -m rvc.modules.model_manager --url "https://example.com/model.zip" --model-name "my_voice"
  
  # Download with PolUVR processing
  python3 -m rvc.modules.model_manager --url "https://example.com/model.zip" --model-name "my_voice" --use-poluvr --poluvr-model "UVR-MDX-NET 1 2 3"
  
  # List available models
  python3 -m rvc.modules.model_manager --list
  
  # Verify model integrity
  python3 -m rvc.modules.model_manager --verify "my_voice"
        """
    )
    
    parser.add_argument("--url", help="URL to download model from")
    parser.add_argument("--model-name", "-n", help="Name for the model directory")
    parser.add_argument("--use-poluvr", action="store_true", help="Apply PolUVR processing to audio files in model")
    parser.add_argument("--poluvr-model", default="UVR-MDX-NET 1 2 3", help="PolUVR model to use for processing")
    parser.add_argument("--list", action="store_true", help="List all available models")
    parser.add_argument("--verify", help="Verify integrity of specified model")
    parser.add_argument("--output-dir", default=rvc_models_dir, help="Output directory for models")
    
    return parser


def verify_model_integrity(model_name):
    """Verify the integrity of a downloaded model"""
    model_path = os.path.join(rvc_models_dir, model_name)
    
    if not os.path.exists(model_path):
        return f"❌ Model '{model_name}' not found in {rvc_models_dir}"
    
    # Check for required files
    required_files = []
    for file in os.listdir(model_path):
        if file.endswith('.pth'):
            file_path = os.path.join(model_path, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            required_files.append(f"  📄 {file} ({size_mb:.1f} MB)")
    
    if not required_files:
        return f"❌ No .pth files found in model '{model_name}'"
    
    # Check for optional index file
    index_files = [f for f in os.listdir(model_path) if f.endswith('.index')]
    index_info = ""
    if index_files:
        index_size = os.path.getsize(os.path.join(model_path, index_files[0])) / (1024 * 1024)
        index_info = f"\n  📊 Index: {index_files[0]} ({index_size:.1f} MB)"
    
    return f"""✅ Model '{model_name}' integrity check:
  📦 Directory: {model_path}
  📁 Status: Valid
  📄 Model files:
{chr(10).join(required_files)}
{index_info}
  🎯 Ready for use!"""


def list_models():
    """List all available models with their status"""
    if not os.path.exists(rvc_models_dir):
        return "📁 No models directory found"
    
    models = []
    for item in os.listdir(rvc_models_dir):
        item_path = os.path.join(rvc_models_dir, item)
        if os.path.isdir(item_path):
            files = os.listdir(item_path)
            has_model = any(f.endswith('.pth') for f in files)
            has_index = any(f.endswith('.index') for f in files)
            
            status = "🟢 Ready" if has_model else "⚠️  Incomplete"
            if has_index:
                status += " + Index"
            
            model_info = f"  {item}: {status}"
            models.append(model_info)
    
    if not models:
        return "📦 No models installed yet\n💡 Download a model using: python3 -m rvc.modules.model_manager --url <url> --model-name <name>"
    
    return f"🎭 Available RVC Models ({len(models)} total):\n" + "\n".join(models)


# Main function for enhanced command-line execution
def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # List models
    if args.list:
        print(list_models())
        return
    
    # Verify model
    if args.verify:
        print(verify_model_integrity(args.verify))
        return
    
    # Download model from URL
    if args.url and args.model_name:
        try:
            if args.use_poluvr:
                print(f"🎛️  PolUVR processing enabled with model: {args.poluvr_model}")
            
            result = enhanced_download_from_url(
                args.url, 
                args.model_name, 
                use_poluvr=args.use_poluvr,
                poluvr_model=args.poluvr_model
            )
            print(result)
            
            # Verify the downloaded model
            if args.verify:
                print(verify_model_integrity(args.model_name))
            
        except gr.Error as e:
            print(f"❌ Error: {str(e)}")
            sys.exit(1)
    else:
        parser.print_help()
        if not args.url and not args.model_name:
            print("\n💡 Quick start:")
            print("  python3 -m rvc.modules.model_manager --url <url> --model-name <name>")
            print("  python3 -m rvc.modules.model_manager --list")


# Legacy function for backward compatibility
def download_from_url(url, dir_name, progress=gr.Progress()):
    """Legacy function for backward compatibility"""
    return enhanced_download_from_url(url, dir_name, progress)


# Legacy function for backward compatibility  
def upload_zip_file(zip_path, dir_name, progress=gr.Progress()):
    """Legacy function for backward compatibility"""
    return enhanced_upload_zip_file(zip_path, dir_name, progress)


# Legacy function for backward compatibility
def upload_separate_files(pth_file, index_file, dir_name, progress=gr.Progress()):
    """Legacy function for backward compatibility"""
    return enhanced_upload_separate_files(pth_file, index_file, dir_name, progress)


if __name__ == "__main__":
    main()
