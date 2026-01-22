import kagglehub
import shutil
import os
import sys

def setup_dataset():
    print("⬇️  Downloading CIFAKE dataset using kagglehub...")
    print("   (This may take a moment depending on your internet connection)")
    
    try:
        # Download dataset
        # This downloads to the local cache directory
        path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
        print(f"✅ Dataset downloaded to cache: {path}")
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("Please ensure you have authenticated with Kaggle if required, or checking your connection.")
        return

    # Define target paths
    # We want to move contents to 'data/train'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_data_dir = os.path.join(base_dir, 'data')
    target_train_dir = os.path.join(target_data_dir, 'train')
    
    # 1. Inspect the downloaded path to understand structure
    # Expected: path/train/REAL, path/train/FAKE
    source_train = os.path.join(path, 'train')
    source_test = os.path.join(path, 'test')

    if not os.path.exists(source_train):
        print(f"❌ Unexpected dataset structure. Could not find 'train' folder in {path}")
        print("   Found:", os.listdir(path))
        return

    # 2. Setup Data Directory
    print(f"📂 Setting up data directory at: {target_data_dir}")
    
    # Remove existing train dir if it exists to ensure clean state
    if os.path.exists(target_train_dir):
        print("   Removing existing 'data/train' directory...")
        shutil.rmtree(target_train_dir)
        
    # Copy from cache to project
    print("📦 Copying training data to project folder...")
    shutil.copytree(source_train, target_train_dir)
    print(f"✅ Training data copied to: {target_train_dir}")
    
    # Optional: Copy test data as 'data/val' or just keep it for later
    # For now, let's keep it simple as the training script splits the train set.
    # But having a separate test set is good.
    target_test_dir = os.path.join(target_data_dir, 'test')
    if os.path.exists(source_test):
         if os.path.exists(target_test_dir):
            shutil.rmtree(target_test_dir)
         shutil.copytree(source_test, target_test_dir)
         print(f"✅ Test data copied to: {target_test_dir}")

    print("\n🎉 Setup Complete!")
    print("   You now have 'data/train' with 'REAL' and 'FAKE' folders.")
    print("   You are ready to run: python models/train_model.py")

if __name__ == "__main__":
    setup_dataset()
