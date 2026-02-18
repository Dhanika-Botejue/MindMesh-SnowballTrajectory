
import cv2
import mediapipe as mp
import math
import numpy as np

class HandThrowDetector:
    def __init__(self, detectionCon=0.7, trackCon=0.7, maxHands=1):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=maxHands,
            min_detection_confidence=0.3, # Even lower to catch blurry frames
            min_tracking_confidence=0.4,
            model_complexity=0 # Lite model is faster and sometimes better with blur
        )
        self.mpDraw = mp.solutions.drawing_utils
        
        # State Management
        self.prev_wrist_coords = None # (x, y)
        self.throw_detected = False
        self.throw_timer = 0 # To keep the "THROWING" text on screen for a bit
        
        # Constants
        self.VELOCITY_THRESHOLD = 40.0 # Pixels per frame. Adjust based on camera resolution/FPS
        self.DIRECTION_THRESHOLD_Y = -15.0 # Significant UPWARD movement (negative Y)
        
        # Colors
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_YELLOW = (0, 255, 255)

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            if draw:
                for handLms in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > handNo:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        # Draw Wrist (0) and Middle Finger Tip (12) larger
                        if id == 0:
                            cv2.circle(img, (cx, cy), 10, self.COLOR_BLUE, cv2.FILLED)
        return lmList

    def detectThrow(self, lmList, img):
        # We need landmarks to calculate velocity
        if not lmList:
            self.prev_wrist_coords = None
            return False, 0

        # Wrist is ID 0
        # lmList format: [id, cx, cy]
        # Find ID 0
        wrist = next((point for point in lmList if point[0] == 0), None)
        if not wrist:
            return False, 0
            
        curr_x, curr_y = wrist[1], wrist[2]
        
        velocity = 0
        delta_y = 0
        
        if self.prev_wrist_coords is not None:
            prev_x, prev_y = self.prev_wrist_coords
            
            # Calculate Velocity (Euclidean distance per frame)
            delta_x = curr_x - prev_x
            delta_y = curr_y - prev_y
            velocity = math.hypot(delta_x, delta_y)
            
            # Throw Detection Logic
            # 1. High Velocity
            # 2. Moving UP (negative delta_y)
            
            check_velocity = velocity > self.VELOCITY_THRESHOLD
            check_direction = delta_y < self.DIRECTION_THRESHOLD_Y 
            
            if check_velocity and check_direction:
                self.throw_detected = True
                self.throw_timer = 15 # Keep text on screen for 15 frames
                # print(f"THROW DETECTED! Velocity: {int(velocity)}, Delta Y: {int(delta_y)}")

            # Visuals: Velocity Vector
            if velocity > 5:
                # Scale arrow for visibility
                end_x = int(curr_x + delta_x * 2)
                end_y = int(curr_y + delta_y * 2)
                cv2.arrowedLine(img, (prev_x, prev_y), (end_x, end_y), self.COLOR_YELLOW, 3)
                cv2.putText(img, f"V:{int(velocity)}", (curr_x + 10, curr_y), 
                            cv2.FONT_HERSHEY_PLAIN, 1, self.COLOR_YELLOW, 2)
                            
        self.prev_wrist_coords = (curr_x, curr_y)
        
        # Handle Timer for display
        is_active = False 
        if self.throw_detected:
            is_active = True
            self.throw_timer -= 1
            if self.throw_timer <= 0:
                self.throw_detected = False
                
        return is_active, velocity

def main():
    cap = cv2.VideoCapture("./snowball-3second-360.mp4")
    # cap = cv2.VideoCapture(0) 
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
        
    detector = HandThrowDetector(detectionCon=0.3, trackCon=0.3)
    
    # User Configuration
    HAND = "RIGHT" # "RIGHT" or "LEFT"

    while True:
        success, img = cap.read()
        if not success:
            break
            
        # ROI Logic
        h, w, c = img.shape
        
        # 1. Vertical Crop: Bottom 50%
        crop_start_y = int(h * 0.35) 
        
        # 2. Horizontal Crop: Split based on HAND
        if HAND == "RIGHT":
            crop_start_x = int(w * 0.4) # Slightly wider than 0.5 to be safe
            crop_end_x = w
        else:
            crop_start_x = 0
            crop_end_x = int(w * 0.6) 
            
        imgROI = img[crop_start_y:h, crop_start_x:crop_end_x]
            
        # Process the ROI
        imgROI = detector.findHands(imgROI)
        lmList = detector.findPosition(imgROI, draw=True)
        
        is_throwing, velocity = detector.detectThrow(lmList, imgROI)
        
        if is_throwing:
            cv2.putText(imgROI, "THROWING", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, detector.COLOR_GREEN, 3)

        # Overlay ROI back onto original image
        img[crop_start_y:h, crop_start_x:crop_end_x] = imgROI
        
        # Draw ROI boundary
        cv2.rectangle(img, (crop_start_x, crop_start_y), (crop_end_x, h), (0, 255, 255), 2)
        cv2.putText(img, "ROI", (crop_start_x, crop_start_y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
