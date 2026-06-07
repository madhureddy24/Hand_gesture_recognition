import cv2
import mediapipe as mp

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)


def count_fingers(hand_landmarks, hand_label):
    fingers = []

    # Thumb
    if hand_label == "Right":
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # Left Hand
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Index, Middle, Ring, Pinky
    tips = [8, 12, 16, 20]

    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def detect_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "Fist ✊"

    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Palm ✋"

    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace ✌️"

    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up 👍"

    else:
        return "Unknown"


while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    total_fingers = 0

    if results.multi_hand_landmarks:

        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # Left / Right Hand Detection
            hand_label = "Unknown"

            if results.multi_handedness:
                hand_label = (
                    results.multi_handedness[hand_idx]
                    .classification[0]
                    .label
                )

            # Finger Counting
            fingers = count_fingers(hand_landmarks, hand_label)
            finger_count = sum(fingers)

            total_fingers += finger_count

            # Gesture Recognition
            gesture = detect_gesture(fingers)

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Bounding Box
            h, w, c = frame.shape

            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)

            cv2.rectangle(
                frame,
                (xmin - 20, ymin - 20),
                (xmax + 20, ymax + 20),
                (255, 0, 255),
                2
            )

            # Hand Label
            cv2.putText(
                frame,
                hand_label,
                (xmin, ymin - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # Gesture
            cv2.putText(
                frame,
                gesture,
                (xmin, ymin - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Finger Count
            cv2.putText(
                frame,
                f"Fingers: {finger_count}",
                (xmin, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

    # Total Fingers
    cv2.putText(
        frame,
        f"Total Fingers: {total_fingers}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Gesture Recognition 2.0", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()