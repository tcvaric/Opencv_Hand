import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img) #with Draw
    #hands, img = detector.findHands(img, flipType=False)#左右反転
    #hands = detector.findHands(img, draw=False) #No Draw
    #print(len(hands))

    #Hand - dict_(lmList - bbox - center - type)

    if hands:
        # Hand1
        hand1 = hands[0]
        lmList1 = hand1["lmList"] #List of 21 Landmarks points
        bbox1 = hand1["bbox"] #Bounding Box info x,y,w,h
        centerPoint1 = hand1["center"] #center of the hand cx,cy
        handType1 = hand1["type"] #Hand Type Left or Right

        #print(len(lmList1),lmList1)
        #print(bbox1)
        #print(centerPoint1)
        #print(handType1)
        fingers1 = detector.fingersUp(hand1)
        # pointの距離を測る
        #length, info, img = detector.findDistance(lmList1[8], lmList1[12], img) #with Draw
        #length, info = detector.findDistance(lmList1[8], lmList1[12]) #No Draw

    if len(hands)==2:
        hand2 = hands[1]
        lmList2 = hand2["lmList"]  # List of 21 Landmarks points
        bbox2 = hand2["bbox"]  # Bounding Box info x,y,w,h
        centerPoint2 = hand2["center"]  # center of the hand cx,cy
        handType2 = hand2["type"]  # Hand Type Left or Right

        fingers2 = detector.fingersUp(hand2)
        #print(handType1, handType2)
        #print(fingers1, fingers2)
        #length, info, img = detector.findDistance(lmList1[16], lmList2[16], img)
        length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()