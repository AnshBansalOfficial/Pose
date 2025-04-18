import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
import os
import json
import pygame.mixer
import random

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Initialize Pygame Mixer for audio
pygame.mixer.init()
success_sound = pygame.mixer.Sound("success.mp3")  # Ensure this file exists
warning_sound = pygame.mixer.Sound("beep.wav")    # Ensure this file exists

# Exercise Templates (Neurorehabilitation-Focused)
EXERCISES = {
    "Mirror Arms": {
        "keypoints": [
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST
        ],
        "target_angle": 170,
        "min_angle": 90,
        "reps": 6,
        "instructions": {
            "en": "Raise both arms symmetrically to shoulder height, hold for 2 seconds.",
            "hi": "दोनों हाथों को कंधे की ऊंचाई तक सममित रूप से उठाएं, 2 सेकंड तक रोकें।"
        },
        "target_zone": (150, 180),
        "goal": "Promotes bilateral coordination and neuroplasticity."
    },
    "Reach and Grab": {
        "keypoints": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
        "target_angle": 160,
        "min_angle": 100,
        "reps": 8,
        "instructions": {
            "en": "Reach forward as if grabbing an object, hold for 2 seconds.",
            "hi": "आगे की ओर बढ़ें जैसे कोई वस्तु पकड़ रहे हों, 2 सेकंड तक रोकें।"
        },
        "target_zone": (140, 170),
        "goal": "Improves arm extension and motor control."
    },
    "Head Tilt": {
        "keypoints": [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
        "target_angle": 30,
        "min_angle": 10,
        "reps": 10,
        "instructions": {
            "en": "Tilt your head side to side slowly, align with shoulders.",
            "hi": "अपने सिर को धीरे-धीरे बगल की ओर झुकाएं, कंधों के साथ संरेखित करें।"
        },
        "target_zone": (20, 40),
        "goal": "Enhances neck mobility and proprioception."
    },
    "Pattern Tracing": {
        "keypoints": [mp_pose.PoseLandmark.RIGHT_WRIST],
        "target_path": [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)],  # Square path
        "reps": 5,
        "instructions": {
            "en": "Trace a square pattern with your left wrist, follow the glowing path.",
            "hi": "अपने बाएं कलाई के साथ एक वर्ग पैटर्न ट्रेस करें, चमकते पथ का पालन करें।"
        },
        "goal": "Boosts cognitive-motor integration and coordination."
    }
}

# Calculate angle between three points
def calculate_angle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * 180 / np.pi
    return angle

# Calculate movement smoothness
def calculate_smoothness(keypoint_history):
    if len(keypoint_history) < 3:
        return 100.0
    velocities = []
    for i in range(1, len(keypoint_history)):
        dx = keypoint_history[i][0] - keypoint_history[i-1][0]
        dy = keypoint_history[i][1] - keypoint_history[i-1][1]
        velocity = np.sqrt(dx**2 + dy**2)
        velocities.append(velocity)
    smoothness = 100 - np.std(velocities) * 100
    return max(0, min(100, smoothness))

# Calculate path tracing accuracy
def calculate_path_accuracy(wrist_pos, target_path, current_segment):
    if not target_path or current_segment >= len(target_path):
        return 100.0
    target_x, target_y = target_path[current_segment]
    distance = np.sqrt((wrist_pos[0] - target_x)**2 + (wrist_pos[1] - target_y)**2)
    accuracy = max(0, 100 - distance * 200)  # Scale distance to 0-100
    return accuracy

# Load leaderboard
def load_leaderboard():
    try:
        with open("data/leaderboard.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Save leaderboard
def save_leaderboard(leaderboard):
    os.makedirs("data", exist_ok=True)
    with open("data/leaderboard.json", "w") as f:
        json.dump(leaderboard, f, indent=2)

# Patient App (Tkinter GUI with OpenCV rendering)
class PatientApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroRehab: Power Recovery")
        self.root.geometry("800x600")
        self.patient_id = "Patient_001"
        self.exercise_data = []
        self.score = 0
        self.level = 1
        self.badges = []
        self.current_exercise = None
        self.current_rep = 0
        self.is_running = False
        self.cap = None
        self.keypoint_history = []
        self.language = "en"
        self.hold_timer = 0
        self.camera_index = 0
        self.combo_count = 0
        self.path_segment = 0
        self.leaderboard = load_leaderboard()

        # GUI Elements
        self.label = tk.Label(root, text="NeuroRehab: Power Your Recovery!", font=("Arial", 24, "bold"), fg="purple")
        self.label.pack(pady=10)

        self.camera_frame = tk.Frame(root)
        self.camera_frame.pack(pady=5)
        tk.Label(self.camera_frame, text="Camera:", font=("Arial", 12)).pack(side=tk.LEFT)
        self.camera_var = tk.StringVar(value="0")
        tk.Radiobutton(self.camera_frame, text="0", variable=self.camera_var, value="0", command=self.update_camera_index).pack(side=tk.LEFT)
        tk.Radiobutton(self.camera_frame, text="1", variable=self.camera_var, value="1", command=self.update_camera_index).pack(side=tk.LEFT)
        tk.Button(self.camera_frame, text="Test Camera", command=self.test_camera, font=("Arial", 12)).pack(side=tk.LEFT)

        self.lang_frame = tk.Frame(root)
        self.lang_frame.pack(pady=5)
        tk.Label(self.lang_frame, text="Language:", font=("Arial", 12)).pack(side=tk.LEFT)
        self.lang_var = tk.StringVar(value="English")
        tk.Radiobutton(self.lang_frame, text="English", variable=self.lang_var, value="English", command=self.toggle_language).pack(side=tk.LEFT)
        tk.Radiobutton(self.lang_frame, text="Hindi", variable=self.lang_var, value="Hindi", command=self.toggle_language).pack(side=tk.LEFT)

        self.exercise_var = tk.StringVar(value="Mirror Arms")
        tk.Label(root, text="Select Exercise:", font=("Arial", 12)).pack()
        exercise_menu = ttk.Combobox(root, textvariable=self.exercise_var, values=list(EXERCISES.keys()), state="readonly", font=("Arial", 12))
        exercise_menu.pack(pady=5)

        self.instruction_label = tk.Label(root, text="", font=("Arial", 14), wraplength=700, justify="center", fg="blue")
        self.instruction_label.pack(pady=10)

        self.feedback_label = tk.Label(root, text="Feedback: Ready to rock!", font=("Arial", 16, "bold"), fg="green")
        self.feedback_label.pack(pady=5)
        self.score_label = tk.Label(root, text=f"Score: {self.score} | Level: {self.level}", font=("Arial", 14))
        self.score_label.pack(pady=5)
        self.badges_label = tk.Label(root, text="Badges: None", font=("Arial", 12))
        self.badges_label.pack(pady=5)
        self.progress = ttk.Progressbar(root, length=500, mode="determinate")
        self.progress.pack(pady=5)

        self.start_button = tk.Button(root, text="Start Exercise", command=self.start_exercise, font=("Arial", 14), bg="green", fg="white")
        self.start_button.pack(pady=10)
        self.stop_button = tk.Button(root, text="Stop Exercise", command=self.stop_exercise, state="disabled", font=("Arial", 14), bg="red", fg="white")
        self.stop_button.pack(pady=10)

        self.update_instructions()
        self.test_camera()

    def update_camera_index(self):
        self.camera_index = int(self.camera_var.get())
        self.test_camera()

    def test_camera(self):
        print(f"Testing camera at index {self.camera_index}...")
        cap = cv2.VideoCapture(self.camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera at index {self.camera_index} is working. Frame shape: {frame.shape}")
            else:
                print(f"Camera at index {self.camera_index} failed to capture frame.")
            cap.release()
        else:
            print(f"Camera at index {self.camera_index} failed to open.")
            cap.release()
            messagebox.showwarning("Camera Test", f"Index {self.camera_index} failed. Check permissions or try index 1.")

    def toggle_language(self):
        self.language = "hi" if self.lang_var.get() == "Hindi" else "en"
        self.update_instructions()

    def update_instructions(self):
        exercise = self.exercise_var.get()
        if exercise in EXERCISES:
            self.instruction_label.config(text=f"{EXERCISES[exercise]['instructions'][self.language]}\nGoal: {EXERCISES[exercise]['goal']}")

    def start_exercise(self):
        if not self.is_running:
            self.current_exercise = self.exercise_var.get()
            if self.current_exercise not in EXERCISES:
                messagebox.showerror("Error", "Please select a valid exercise.")
                return
            self.current_rep = 0
            self.exercise_data = []
            self.keypoint_history = []
            self.hold_timer = 0
            self.combo_count = 0
            self.path_segment = 0
            if self.cap:
                self.cap.release()
                self.cap = None
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Failed to open camera at index {self.camera_index}.")
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"Camera opened: {self.cap.isOpened()}, Resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
            self.is_running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            # Show animated tutorial (simulated with text for now)
            messagebox.showinfo("Tutorial", f"Follow the glowing path for {self.current_exercise}!\n{EXERCISES[self.current_exercise]['instructions'][self.language]}")
            threading.Thread(target=self.update_frame, daemon=True).start()

    def stop_exercise(self):
        if self.is_running:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            cv2.destroyAllWindows()
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.feedback_label.config(text="Feedback: Session ended. Great effort!")
            self.save_exercise_data()
            # Update leaderboard
            self.leaderboard.append({"patient_id": self.patient_id, "score": self.score, "date": datetime.now().isoformat()})
            self.leaderboard = sorted(self.leaderboard, key=lambda x: x["score"], reverse=True)[:5]  # Top 5
            save_leaderboard(self.leaderboard)

    def update_frame(self):
        motivational_phrases = [
            ("You're killing it!", "आप इसे शानदार कर रहे हैं!"),
            ("Keep it smooth!", "इसे सुचारू रखें!"),
            ("You're a rehab rockstar!", "आप एक रिहैब रॉकस्टार हैं!"),
            ("Power through!", "शक्ति के साथ आगे बढ़ें!")
        ]
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                print("Failed to capture frame.")
                self.feedback_label.config(text="Feedback: Camera feed lost.")
                self.stop_exercise()
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            feedback_text = "No pose detected"
            combo_text = f"Combo: {self.combo_count}x" if self.combo_count > 1 else ""
            power_level = 0
            wrist_x, wrist_y = None, None  # Initialize to None

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=3),  # Neon magenta
                                         mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3))  # Neon cyan
                exercise = EXERCISES[self.current_exercise]
                color = (0, 0, 255)

                if self.current_exercise == "Pattern Tracing":
                    wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                    wrist_x, wrist_y = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])
                    accuracy = calculate_path_accuracy([wrist.x, wrist.y], exercise["target_path"], self.path_segment)
                    power_level = accuracy
                    self.keypoint_history.append([wrist.x, wrist.y])
                    smoothness = calculate_smoothness(self.keypoint_history[-30:])
                    if accuracy > 90:
                        self.hold_timer += 1
                        feedback_text = "On target! Keep tracing!" if self.language == "en" else "लक्ष्य पर! ट्रेसिंग जारी रखें!"
                        color = (0, 255, 0)
                        success_sound.play()
                        if self.hold_timer >= 20:
                            self.path_segment += 1
                            self.hold_timer = 0
                            self.score += 20
                            self.combo_count += 1
                            if self.path_segment >= len(exercise["target_path"]):
                                self.current_rep += 1
                                self.path_segment = 0
                                self.keypoint_history = []
                    else:
                        feedback_text = "Follow the glowing path!" if self.language == "en" else "चमकते पथ का पालन करें!"
                        color = (0, 165, 255)
                        warning_sound.play()
                    # Draw target path
                    for i, (tx, ty) in enumerate(exercise["target_path"]):
                        px, py = int(tx * frame.shape[1]), int(ty * frame.shape[0])
                        cv2.circle(frame, (px, py), 10, (0, 255, 0) if i == self.path_segment else (255, 255, 255), -1)
                        if i > 0:
                            prev_x, prev_y = int(exercise["target_path"][i-1][0] * frame.shape[1]), int(exercise["target_path"][i-1][1] * frame.shape[0])
                            cv2.line(frame, (prev_x, prev_y), (px, py), (0, 255, 0), 2)
                else:
                    keypoints = [results.pose_landmarks.landmark[kp] for kp in exercise["keypoints"]]
                    if self.current_exercise == "Mirror Arms":
                        left_angle = calculate_angle(keypoints[0], keypoints[1], keypoints[2])
                        right_angle = calculate_angle(keypoints[3], keypoints[4], keypoints[5])
                        angle = (left_angle + right_angle) / 2
                        symmetry = 100 - abs(left_angle - right_angle) * 2
                        power_level = symmetry
                    else:
                        angle = calculate_angle(keypoints[0], keypoints[1], keypoints[2])
                        power_level = min(100, (angle - exercise["min_angle"]) / (exercise["target_angle"] - exercise["min_angle"]) * 100)
                    wrist_x, wrist_y = int(keypoints[2].x * frame.shape[1]), int(keypoints[2].y * frame.shape[0])
                    self.keypoint_history.append([keypoints[2].x, keypoints[2].y])
                    smoothness = calculate_smoothness(self.keypoint_history[-30:])

                    target_zone = exercise["target_zone"]
                    if angle >= target_zone[0] and angle <= target_zone[1]:
                        self.hold_timer += 1
                        feedback_text = random.choice(motivational_phrases)[0 if self.language == "en" else 1]
                        color = (0, 255, 0)
                        success_sound.play()
                        if self.hold_timer >= 20:
                            self.current_rep += 1
                            self.score += 10 + self.combo_count * 5
                            self.combo_count += 1
                            self.hold_timer = 0
                            self.keypoint_history = []
                    elif angle >= exercise["min_angle"]:
                        feedback_text = "Almost there! Extend more!" if self.language == "en" else "लगभग हो गया! और फैलाएं!"
                        color = (0, 165, 255)
                        self.combo_count = 0
                        warning_sound.play()
                    else:
                        feedback_text = "Push harder! Lift up!" if self.language == "en" else "जोर से धक्का दें! ऊपर उठाएं!"
                        color = (0, 0, 255)
                        self.combo_count = 0
                        warning_sound.play()

                # Draw glowing target zone only if wrist coordinates are valid
                if wrist_x is not None and wrist_y is not None:
                    radius = 50 + int(self.hold_timer * 2)  # Grow with hold time
                    cv2.ellipse(frame, (wrist_x, wrist_y), (radius, radius), 0, 0, 360, color, 2)
                    cv2.circle(frame, (wrist_x, wrist_y), 10, (255, 255, 0), -1)  # Highlight wrist

                # Draw power bar
                power_bar_width = int(power_level * 3)
                cv2.rectangle(frame, (10, frame.shape[0] - 30), (10 + power_bar_width, frame.shape[0] - 10), (0, 255, 0), -1)
                cv2.rectangle(frame, (10, frame.shape[0] - 30), (310, frame.shape[0] - 10), (255, 255, 255), 2)
                cv2.putText(frame, "Power", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Display metrics
                metrics_y = 30
                cv2.putText(frame, f"Angle: {angle:.1f}°" if self.current_exercise != "Pattern Tracing" else f"Accuracy: {accuracy:.1f}%", 
                           (10, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                metrics_y += 30
                cv2.putText(frame, f"Smoothness: {smoothness:.1f}%", (10, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                metrics_y += 30
                cv2.putText(frame, feedback_text, (10, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                metrics_y += 30
                cv2.putText(frame, f"Rep: {self.current_rep}/{exercise['reps']}", (10, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                metrics_y += 30
                cv2.putText(frame, combo_text, (10, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Draw progress wheel
                progress = min(self.current_rep / exercise["reps"], 1.0)
                center = (frame.shape[1] - 50, 50)
                cv2.circle(frame, center, 40, (0, 0, 0), -1)
                cv2.circle(frame, center, 40, (0, 255, 0), 5)
                cv2.circle(frame, center, 38, (0, 0, 0), -1)
                angle = int(360 * progress)
                cv2.ellipse(frame, center, (38, 38), 0, 0, angle, (255, 255, 0), -1)

                # Save data to memory (no local storage)
                self.exercise_data.append({
                    "timestamp": datetime.now().isoformat(),
                    "exercise": self.current_exercise,
                    "rep": self.current_rep,
                    "angle": angle if self.current_exercise != "Pattern Tracing" else accuracy,
                    "smoothness": smoothness,
                    "power": power_level
                })

                # Check completion
                if self.current_rep >= exercise["reps"]:
                    self.score += 50
                    badge = f"{self.current_exercise} Master"
                    if badge not in self.badges:
                        self.badges.append(badge)
                        messagebox.showinfo("Badge Earned!", f"New Badge: {badge}")
                    self.level = 1 + self.score // 100
                    feedback_text = "Exercise completed! You're unstoppable!" if self.language == "en" else "व्यायाम पूरा हुआ! आप रुक नहीं सकते!"
                    self.feedback_label.config(text=feedback_text)
                    self.stop_exercise()
                    messagebox.showinfo("Victory!", f"Exercise completed!\nScore: {self.score}\nLevel: {self.level}\nBadges: {', '.join(self.badges)}")

            # Display frame
            cv2.imshow("NeuroRehab: Power Recovery", frame)

            # Update GUI
            self.root.after(0, self.update_gui, feedback_text, self.current_rep)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop_exercise()
                break

            time.sleep(0.08)  # ~12.5 FPS for smooth visuals

    def update_gui(self, feedback_text, current_rep):
        if self.is_running:
            self.feedback_label.config(text=f"Feedback: {feedback_text}")
            self.score_label.config(text=f"Score: {self.score} | Level: {self.level}")
            self.badges_label.config(text=f"Badges: {', '.join(self.badges) or 'None'}")
            if self.current_exercise:
                exercise = EXERCISES[self.current_exercise]
                self.progress["value"] = (current_rep / exercise["reps"]) * 100

    def save_exercise_data(self):
        if self.exercise_data:
            self.feedback_label.config(text="Feedback: Session completed!" if self.language == "en" else "सत्र पूरा हुआ!")

# Main Execution
if __name__ == "__main__":
    root = tk.Tk()
    app = PatientApp(root)
    root.mainloop()