import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("G:/PBL 2/Model/keras_model.h5", "G:/PBL 2/Model/labels.txt")
offset = 20
imgSize = 300
labels = ["Hello", "I Love You", "No", "Okey", "Thanks", "Yes"]

while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to read from camera")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Safe crop limits
        x_start = max(x - offset, 0)
        y_start = max(y - offset, 0)
        x_end = min(x + w + offset, img.shape[1])
        y_end = min(y + h + offset, img.shape[0])

        imgCrop = img[y_start:y_end, x_start:x_end]

        if imgCrop.size == 0:
            print("⚠️ Empty image after cropping, skipping frame")
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        except Exception as e:
            print("⚠️ Prediction failed:", e)
            continue

        # Draw results
        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset - 10), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # Show cropped and white background images
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    # Show main webcam feed
    cv2.imshow('Image', imgOutput)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
