import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

def verify():
    print("=" * 60)
    print("RUNNING ENVIRONMENT AND SETUP VERIFICATION")
    print("=" * 60)

    # 1. Check folder structure
    base_dir = Path(__file__).resolve().parent
    stage1_dir = base_dir / "models" / "stage1"
    stage2_bundle = base_dir / "models" / "stage2" / "stage2_text_bundle.joblib"
    best_model_pt = stage1_dir / "best_model.pt"
    best_model_info = stage1_dir / "best_model_info.json"

    print("\n[1] Checking Directory Structure:")
    for path in [stage1_dir, best_model_pt, best_model_info, stage2_bundle]:
        exists = path.exists()
        status = "EXISTS" if exists else "MISSING"
        rel_path = str(path.relative_to(base_dir)).replace("\\", "/")
        print(f" - {rel_path}: {status}")
        if not exists:
            print("ERROR: Missing required files.")
            sys.exit(1)

    # 2. Try importing requirements
    print("\n[2] Checking Dependencies:")
    dependencies = [
        "torch", "torchvision", "timm", "ultralytics", 
        "sentence_transformers", "sklearn", "pandas", "numpy", "scipy", "joblib", "PIL"
    ]
    all_ok = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f" - {dep}: Installed")
        except ImportError:
            print(f" - {dep}: NOT INSTALLED")
            all_ok = False

    if not all_ok:
        print("\nERROR: Some dependencies are missing. Run: pip install -r requirements.txt")
        sys.exit(1)

    # 3. Test Loading and Inference
    print("\n[3] Testing Model Loading & Inference Pipeline:")
    try:
        from pipeline import MaintenancePipeline
        print(" - Importing pipeline.py: OK")
        
        print(" - Loading MaintenancePipeline (this may take a few seconds)...")
        prod_pipeline = MaintenancePipeline(base_dir=base_dir)
        print(" - Loading pipeline: Loaded successfully!")

        print(" - Running dummy prediction on the pipeline...")
        # Create a dummy image
        dummy_img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy_img_array)
        dummy_text = "الحنفية بتسرب ميه"

        # Test pipeline system
        prod_output = prod_pipeline.predict(dummy_image, dummy_text)
        print("\npipeline prediction output:")
        print(f" - Image Prediction Class: {prod_output['image_class']}")
        print(f" - Image Prediction Confidence: {prod_output['image_confidence']:.4f}")
        print(f" - Text Prediction Category: {prod_output['text_category']}")
        print(f" - Text Prediction Confidence: {prod_output['text_confidence']:.4f}")
        print(f" - Final Decision: {prod_output['decision']}")
        print(f" - Explanation: {prod_output['explanation_ar']}".encode('utf-8', errors='ignore').decode('cp1252', errors='ignore'))

        print("\nVerification PASSED! The pipeline is fully operational.")

    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify()

