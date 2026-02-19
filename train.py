from ultralytics import YOLO

def main():
    # 1. Load the base model
    # We use 'n' (nano) for the highest speed during your snowball throw
    model = YOLO("yolov8n.pt") 

    # 2. Start Training
    model.train(
        data="hand-throw/data.yaml",      # Path to your dataset config
        epochs=50,             # Number of passes through the data
        imgsz=640,             # Square size of images (standard)
        batch=16,              # Number of images per "bite" (Adjust based on VRAM)
        device=0,              # Force use of RTX 3060 Ti
        workers=0,             # Use 0 for Windows to avoid multiprocessing issues
        amp=False,             # Disable mixed precision for stability
        project="hand-throw", # Main folder name
        name="HandV1",         # Sub-folder for this specific run
        exist_ok=True          # Overwrite if you run it again with the same name
    )

if __name__ == "__main__":
    main()