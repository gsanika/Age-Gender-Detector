
import cv2
from PIL import Image
from age_and_gender import AgeAndGender

# Load the model once
predictor = AgeAndGender()

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam. Check camera permissions.")
    raise SystemExit

print("Webcam started. Press Q to quit.")

frame_count = 0
results = []

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    frame_count += 1

    # Run prediction every 5 frames for better speed
    if frame_count % 5 == 1:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        try:
            results = predictor.predict(image)
        except Exception as e:
            print("Prediction error:", e)
            break

    # Draw face boxes and predictions
    for result in results:
        left, top, right, bottom = result["face"]

        age = result["age"]["value"]
        gender = result["gender"]["value"]

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        label = f"Age: {age} | Gender: {gender}"

        cv2.putText(
            frame,
            label,
            (left, max(top - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Real-Time Age and Gender Detector", frame)

    # Press Q to close
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()