import tempfile
import cv2
import numpy as np
from pathlib import Path

def generate_test_video(
    duration_seconds: int = 5,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    complexity: str = "low"
) -> Path:
    """
    Generates a synthetic UI recording video to test the pipeline.
    
    complexity="low": Minimal changes, single colored box.
    complexity="high": Rapid changes, lots of simulated UI elements, text, and colors,
                       designed to trigger YOLO, OCR, and DOM nesting algorithms frequently.
    """
    path = Path(tempfile.mkdtemp()) / f"stress_test_video_{complexity}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    total_frames = duration_seconds * fps
    
    for i in range(total_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
        
        if complexity == "low":
            # Just a moving blue box
            x = int((i / total_frames) * (width - 100))
            cv2.rectangle(frame, (x, 100), (x + 100, 200), (255, 0, 0), -1)
            cv2.putText(frame, "Low Load", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
        elif complexity == "high":
            # Simulate a dense UI with lots of boxes and text
            # Change layout every 1 second (fps frames) to trigger many key states
            layout_state = i // fps
            
            # Header
            cv2.rectangle(frame, (0, 0), (width, 50), (200, 200, 200), -1)
            cv2.putText(frame, f"High Load - State {layout_state}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Simulated complex grid of elements
            cols = 4
            rows = 5
            cell_w = width // cols
            cell_h = (height - 50) // rows
            
            for r in range(rows):
                for c in range(cols):
                    # Change color based on state to ensure Module A detects it
                    color_val = (100 + (layout_state * 10 + r * 20 + c * 30) % 155)
                    color = (color_val, color_val, 255) if (r + c) % 2 == 0 else (255, color_val, color_val)
                    
                    x1 = c * cell_w + 10
                    y1 = 50 + r * cell_h + 10
                    x2 = x1 + cell_w - 20
                    y2 = y1 + cell_h - 20
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                    
                    # Add OCR text targets inside boxes
                    cv2.putText(frame, f"B({r},{c})", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Add inner simulated elements (nesting for Module C)
                    cv2.rectangle(frame, (x1 + 10, y1 + 40), (x2 - 10, y2 - 10), (255, 255, 255), -1)
                    cv2.putText(frame, f"Val:{layout_state}", (x1 + 15, y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        writer.write(frame)
        
    writer.release()
    return path
