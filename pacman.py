import cv2
import numpy as np
import random
import time

cap = cv2.VideoCapture(0)

running = False
score = 0
start_time = None

def generate_food():
    num_points = random.randint(8, 20)
    points = [(random.randint(0, 640), random.randint(0, 480)) for _ in range(num_points)]
    return points

food_points = generate_food()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    key_pose = (random.randint(0, 640), random.randint(0, 480))

    if running:
        new_food_points = []
        for (x, y) in food_points:
            if (x - key_pose[0])**2 + (y - key_pose[1])**2 > 400:
                new_food_points.append((x, y))
            else:
                score += 1
        food_points = new_food_points
        if not food_points:
            food_points = generate_food()

    for (x, y) in food_points:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        running = True
        start_time = time.time()
        score = 0
        food_points = generate_food()
    elif key == ord('d'):
        running = False

    if running and (time.time() - start_time) >= 60:
        running = False
        print(f'Final Score: {score}')

cap.release()
cv2.destroyAllWindows()

