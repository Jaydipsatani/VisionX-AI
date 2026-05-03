import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import pyautogui


# =========================
# Configuration
# =========================
wCam, hCam = 640, 480
frameR = 100
smoothening = 7

# =========================
# Variables
# =========================
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

clickDelay = 0.3
rightClickDelay = 0.6
scrollDelay = 0.05
dragActive = False
lastClickTime = 0
lastRightClickTime = 0
lastScrollTime = 0

# =========================
# Camera Setup
# =========================
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

# =========================
# Main Loop
# =========================
while True:
    success, img = cap.read()
    if not success:
        continue

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]   # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        fingers = detector.fingersUp()

        # Draw active frame
        cv2.rectangle(img, (frameR, frameR),
                      (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

        currentTime = time.time()

        # =========================
        # Mouse Move (Index Only)
        # =========================
        if fingers == [0, 1, 0, 0, 0]:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            autopy.mouse.move(wScr - clocX, clocY)
            plocX, plocY = clocX, clocY

            cv2.circle(img, (x1, y1), 12, (255, 0, 255), cv2.FILLED)

        # =========================
        # Left Click (Index + Middle close)
        # =========================
        elif fingers == [0, 1, 1, 0, 0]:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length < 40 and (currentTime - lastClickTime) > clickDelay:
                autopy.mouse.click()
                lastClickTime = currentTime
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 12,
                           (0, 255, 0), cv2.FILLED)

        # =========================
        # Right Click (Middle Only)
        # =========================
        elif fingers == [0, 0, 1, 0, 0]:
            if (currentTime - lastRightClickTime) > rightClickDelay:
                autopy.mouse.click(button=autopy.mouse.Button.RIGHT)
                lastRightClickTime = currentTime

      
        # =========================
        # Scroll (Index + Middle + Ring)
        # =========================
        elif fingers == [0, 1, 1, 1, 0]:
            if (currentTime - lastScrollTime) > scrollDelay:
                if y1 < hCam // 3:
                    pyautogui.scroll(250)      # Scroll Up
                elif y1 > 2 * hCam // 3:
                    pyautogui.scroll(-250)     # Scroll Down
                lastScrollTime = currentTime    

        # =========================
        # Drag (Fist)
        # =========================
        elif fingers == [0, 0, 0, 0, 0]:
            if not dragActive:
                autopy.mouse.toggle(autopy.mouse.Button.LEFT, True)
                dragActive = True

        # =========================
        # Release Drag (Open Hand)
        # =========================
        elif fingers == [1, 1, 1, 1, 1]:
            if dragActive:
                autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)
                dragActive = False

    # =========================
    # FPS Calculation
    # =========================
    cTime = time.time()
    fps = int(1 / (cTime - pTime)) if cTime != pTime else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {fps}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2)

    # =========================
    # Display
    # =========================
    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
