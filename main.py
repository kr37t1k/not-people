import cv2
import numpy as np
from collections import defaultdict
import time
import threading
from queue import Queue


class PeopleCounter:
    def __init__(
            self,
            video_source: int | str,
            line_position: float = 0.5,
    ):
        """
        args:
            video_source: источник видео
            line_position: позиция линии подсчета
            skip_frames: пропуск кадров для ускорения
        """
        self.cap = cv2.VideoCapture(video_source)

        self.frame_count = 0
        self.last_processed_frame = None
        self.frame_queue = Queue(maxsize=10)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        self.line_position = line_position
        self.count_line = None

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.people_count = 0
        self.tracked_people = {}
        self.next_id = 0
        self.crossed_ids = set()

        self.running = True

    def setup_counting_line(self, frame_height):
        self.count_line = int(frame_height * self.line_position)
        return self.count_line

    def detect_people(self, frame):
        scale_factor = 0.5
        small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

        boxes, weights = self.hog.detectMultiScale(
            small_frame,
            winStride=(16, 16),
            padding=(16, 16),
            scale=1.1,
            hitThreshold=0.3
        )

        people = []
        for (x, y, w, h) in boxes:
            x = int(x / scale_factor)
            y = int(y / scale_factor)
            w = int(w / scale_factor)
            h = int(h / scale_factor)
            if w > 25 and h > 200:
                center_x = x + w // 2
                center_y = y + h // 2
                people.append({
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y)
                })

        return people

    def track_people(self, current_people):
        current_time = time.time()
        updated_tracks = {}

        for person in current_people:
            center = person['center']
            best_match_id = None
            min_distance = 10

            for person_id, track_data in self.tracked_people.items():
                last_center = track_data['center']
                distance = np.sqrt((center[0] - last_center[0]) ** 2 +
                                   (center[1] - last_center[1]) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    best_match_id = person_id

            if best_match_id is not None:
                updated_tracks[best_match_id] = {
                    'center': center,
                    'bbox': person['bbox'],
                    'last_seen': current_time
                }
            else:
                updated_tracks[self.next_id] = {
                    'center': center,
                    'bbox': person['bbox'],
                    'last_seen': current_time
                }
                self.next_id += 1

        self.tracked_people = {
            k: v for k, v in updated_tracks.items()
            if current_time - v['last_seen'] < 3.0
        }

        return self.tracked_people

    def check_crossing(self, tracks, frame_height):
        if self.count_line is None:
            self.setup_counting_line(frame_height)

        for person_id, track_data in tracks.items():
            center_y = track_data['center'][1]

            if (person_id not in self.crossed_ids and
                    center_y > self.count_line):
                self.people_count += 1
                self.crossed_ids.add(person_id)
                print(f"Всего: {self.people_count}")

    def read_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Пропускаем кадры для ускорения
            self.frame_count += 1
            if self.frame_count != 0:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)

        self.running = False

    def process_frames(self):
        while self.running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=1)

                people = self.detect_people(frame)

                # Быстрое отслеживание
                tracks = self.track_people(people)

                # Проверка пересечения
                self.check_crossing(tracks, frame.shape[0])

                # Отрисовка
                frame = self.draw_ui(frame, tracks)

                # Показать кадр
                cv2.imshow('People Counter', frame)

                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.people_count = 0
                    self.crossed_ids.clear()

            except:
                continue

    def draw_ui(self, frame, tracks):
        frame_height = frame.shape[0]

        if self.count_line is None:
            self.setup_counting_line(frame_height)

        cv2.line(frame, (0, self.count_line),
                 (frame.shape[1], self.count_line), (0, 255, 0), 2)

        for track_data in tracks.values():
            x, y, w, h = track_data['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f'PeopleCount: {self.people_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, 'Q - exit, R - reset counter',
                    (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def run(self):
        print("Q - выход, R - сброс")

        reader_thread = threading.Thread(target=self.read_frames)
        reader_thread.daemon = False
        reader_thread.start()

        self.process_frames()

        self.release()

    def release(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Итоговое количество людей: {self.people_count}")


if __name__ == "__main__":
    print("Оптимизированная детекция людей ")
    counter = PeopleCounter(video_source="example_video_camera.mp4",
                            line_position=0.5)
    counter.run()
