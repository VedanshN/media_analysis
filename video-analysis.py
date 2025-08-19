# --- (Keep your imports and CONFIGURATION the same) ---
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import transcription

# --- CONFIGURATION ---
VIDEO_PATH = "test.mp4"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
FRAMES_TO_EXTRACT = 30  # Analyze a frame every 30 frames
MIN_BRIGHTNESS = 50
MIN_FOCUS = 100
TOP_N_FRAMES = 1

def analyze_frame(frame, face_cascade, yolo_model):
    """
    Analyzes a single video frame.
    """
    # 1. Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Brightness Analysis
    brightness = gray.mean()

    # 3. Focus/Blur Analysis
    focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 4. Face Detection
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    num_faces = len(faces)

    # 5. Object Detection
    results = yolo_model(frame, verbose=False)
    result = results[0]
    
    num_objects = 0 
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]
        if class_name != "person":
            num_objects += 1

    return {
        "brightness": brightness,
        "focus": focus_measure,
        "faces": num_faces,
        "frame_data": frame,
        "objects": num_objects,
    }

def score_frame(analysis_results):
    """
    Scores a frame based on its analysis results.
    """
    if analysis_results["brightness"] < MIN_BRIGHTNESS or analysis_results["focus"] < MIN_FOCUS:
        return 0
    
    score = 0
    score += analysis_results["focus"]
    score += analysis_results["faces"] * 100
    score += analysis_results["objects"] * 25
    return score

def main():
    """
    Main function to run the video analysis.
    """
    print("Starting video analysis...")

    # --- VALIDATION ---
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at '{VIDEO_PATH}'")
        return
    if not os.path.exists(CASCADE_PATH):
        print(f"Error: Haar Cascade file not found at '{CASCADE_PATH}'")
        return

    try:
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        yolo_model = YOLO('yolov11n.pt')
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Warning: Could not determine video FPS. Assuming 30.")
        fps = 30

    print("startin transcription...")
    try:
        id_segments = transcription.transcription()
    except Exception as e:
        print(f"Error getting transcription data: {e}")
        cap.release()
        return


    for i, segment in enumerate(id_segments):
        start_time = segment[1][0]
        end_time = segment[2][0]
        
        # Convert times to frame numbers
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        print(f"\n--- Analyzing Segment {i+1} (Frames {start_frame} to {end_frame}) ---")
        
        analyzed_frames = []
        
        # Set the video capture to the start frame of the segment
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame_num = start_frame
        while current_frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break # End of video
            
            # Analyze every Nth frame within the segment
            if (current_frame_num - start_frame) % FRAMES_TO_EXTRACT == 0:
                print(f"Analyzing frame {current_frame_num}...")
                analysis = analyze_frame(frame, face_cascade, yolo_model)
                score = score_frame(analysis)
                analyzed_frames.append({
                    "frame_number": current_frame_num,
                    "score": score,
                    "analysis": analysis
                })

            current_frame_num += 1

        # --- RANKING AND SELECTION for the current segment ---
        if not analyzed_frames:
            print("No suitable frames were found in this segment.")
            continue

        sorted_frames = sorted(analyzed_frames, key=lambda x: x["score"], reverse=True)
        top_frames = sorted_frames[:TOP_N_FRAMES]

        print("\n--- Top Frame(s) for this Segment ---")
        for j, result in enumerate(top_frames):
            analysis = result["analysis"]
            print(
                f"#{j+1}: Frame {result['frame_number']} | Score: {result['score']:.2f} | "
                f"Brightness: {analysis['brightness']:.2f} | Focus: {analysis['focus']:.2f} | "
                f"Faces: {analysis['faces']}"
            )
            # Save the best frames with a unique name per segment
            output_filename = f"best_frame_segment_{i+1}_rank_{j+1}.jpg"
            cv2.imwrite(output_filename, result["analysis"]["frame_data"])
            print(f" -> Saved as {output_filename}")

    # --- FIX: Release capture object AFTER all loops are done ---
    cap.release()
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()