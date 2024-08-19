import cv2
import dlib
import time
import math
import mysql.connector
from pytesseract import image_to_string
from datetime import datetime


# SQL Database Setup
def setup_database():
    conn = mysql.connector.connect(
        host='localhost',
        user='your_username',
        password='your_password',
        database='vehicles_db'
    )
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicles (
            id INT AUTO_INCREMENT PRIMARY KEY,
            license_plate VARCHAR(255) NOT NULL,
            speed FLOAT NOT NULL,
            timestamp DATETIME NOT NULL
        )
    ''')
    conn.commit()
    return conn, cursor


# Function to insert data into the database
def insert_vehicle_data(cursor, license_plate, speed, timestamp):
    cursor.execute('''
        INSERT INTO vehicles (license_plate, speed, timestamp)
        VALUES (%s, %s, %s)
    ''', (license_plate, speed, timestamp))


# Function to estimate speed
def estimate_speed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 18
    speed = d_meters * fps * 3.6
    return speed


# Load the classifiers
car_cascade = cv2.CascadeClassifier('vehicle.xml')

# Initialize video capture
video = cv2.VideoCapture('carsVideo.mp4')
WIDTH, HEIGHT = 1280, 720

# Initialize database
conn, cursor = setup_database()


# Function to recognize license plates
def recognize_license_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = image_to_string(gray, config='--psm 8')
    return text.strip()


# Function to track and detect vehicles
def track_and_detect_vehicles():
    rectangle_color = (0, 255, 255)
    frame_counter = 0
    current_car_id = 0
    fps = 0
    car_tracker = {}
    car_location1 = {}
    car_location2 = {}
    speed = [None] * 1000
    last_recorded_plates = {}
    out = cv2.VideoWriter('outTraffic.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (WIDTH, HEIGHT))

    while True:
        start_time = time.time()
        rc, image = video.read()
        if image is None:
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        result_image = image.copy()

        frame_counter += 1
        car_id_to_delete = []

        for car_id in car_tracker.keys():
            tracking_quality = car_tracker[car_id].update(image)
            if tracking_quality < 7:
                car_id_to_delete.append(car_id)

        for car_id in car_id_to_delete:
            car_tracker.pop(car_id, None)
            car_location1.pop(car_id, None)
            car_location2.pop(car_id, None)

        if not (frame_counter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = car_cascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h
                match_car_id = None

                for car_id in car_tracker.keys():
                    tracked_position = car_tracker[car_id].get_position()
                    t_x = int(tracked_position.left())
                    t_y = int(tracked_position.top())
                    t_w = int(tracked_position.width())
                    t_h = int(tracked_position.height())
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h
                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (
                            x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        match_car_id = car_id

                if match_car_id is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    car_tracker[current_car_id] = tracker
                    car_location1[current_car_id] = [x, y, w, h]
                    current_car_id += 1

        for car_id in car_tracker.keys():
            tracked_position = car_tracker[car_id].get_position()
            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())
            cv2.rectangle(result_image, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangle_color, 4)
            car_location2[car_id] = [t_x, t_y, t_w, t_h]

        end_time = time.time()
        if end_time != start_time:
            fps = 1.0 / (end_time - start_time)

        for i in car_location1.keys():
            if frame_counter % 1 == 0:
                [x1, y1, w1, h1] = car_location1[i]
                [x2, y2, w2, h2] = car_location2[i]
                car_location1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] is None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = estimate_speed([x1, y1, w1, h1], [x1, y2, w2, h2])

                    if speed[i] is not None and y1 >= 180:
                        cv2.putText(result_image, str(int(speed[i])) + " km/h", (int(x1 + w1 / 2), int(y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 100), 2)

                        # If speed exceeds 60 km/h, recognize the license plate and store details
                        if speed[i] > 60:
                            plate_image = image[y:y + h, x:x + w]
                            license_plate = recognize_license_plate(plate_image)
                            if license_plate:
                                today = datetime.now().strftime('%Y-%m-%d')
                                if license_plate not in last_recorded_plates or last_recorded_plates[
                                    license_plate] != today:
                                    last_recorded_plates[license_plate] = today
                                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    insert_vehicle_data(cursor, license_plate, speed[i], timestamp)
                                    conn.commit()

        cv2.imshow('result', result_image)
        out.write(result_image)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    out.release()
    conn.close()


if __name__ == '__main__':
    track_and_detect_vehicles()
