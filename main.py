import cv2
import numpy as np
from collections import defaultdict
import time


class PeopleCounter:
    def __init__(
            self,
            video_source: int | str,
            line_position: float = 0.5
    ):
        """
        args: video_source: источник видео (0 - камера or "C:/path/to/video.mp4" - видеофайл),
        line_position: позиция линии подсчета.
        """
        self.cap = cv2.VideoCapture(video_source)
        self.line_position = line_position
        self.count_line = None

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.people_count = 0
        self.tracked_people = {}
        self.next_id = 0
        self.crossed_ids = set()
        self.last_seen = defaultdict(float)

    def setup_counting_line(self, frame_height):
        self.count_line = int(frame_height * self.line_position)
        return self.count_line

    def detect_people(self, frame):
        boxes, weights = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(32, 32),
            scale=1.05,
            hitThreshold=0.5
        )

        people = []
        for (x, y, w, h) in boxes:
            # Фильтрация по размеру
            if w > 60 and h > 100:
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
        matched_ids = set()

        for person in current_people:
            center = person['center']
            min_distance = float('inf')
            best_match_id = None

            for person_id, track_data in self.tracked_people.items():
                last_center = track_data['center']
                distance = np.sqrt((center[0] - last_center[0]) ** 2 +
                                   (center[1] - last_center[1]) ** 2)

                if distance < 100 and distance < min_distance and person_id not in matched_ids:
                    min_distance = distance
                    best_match_id = person_id

            if best_match_id is not None:
                updated_tracks[best_match_id] = {
                    'center': center,
                    'bbox': person['bbox'],
                    'last_seen': current_time
                }
                matched_ids.add(best_match_id)
            else:
                updated_tracks[self.next_id] = {
                    'center': center,
                    'bbox': person['bbox'],
                    'last_seen': current_time
                }
                self.next_id += 1

        for person_id in list(self.tracked_people.keys()):
            if current_time - self.tracked_people[person_id]['last_seen'] > 2.0:
                if person_id in updated_tracks:
                    del updated_tracks[person_id]

        self.tracked_people = updated_tracks
        return self.tracked_people

    def check_crossing(self, tracks, frame_height):
        if self.count_line is None:
            self.setup_counting_line(frame_height)

        for person_id, track_data in tracks.items():
            center_y = track_data['center'][1] # or X for horizontal line

            if (person_id not in self.crossed_ids and
                    center_y > self.count_line):

                if person_id in self.last_seen:
                    prev_y = self.last_seen[person_id]
                    if center_y > prev_y:
                        self.people_count += 1
                        self.crossed_ids.add(person_id)
                        print(f"Всего: {self.people_count}")
            self.last_seen[person_id] = center_y

    def draw_ui(self, frame, tracks):
        frame_height = frame.shape[0]

        # Линия подсчета
        if self.count_line is None:
            self.setup_counting_line(frame_height)

        cv2.line(frame, (0, self.count_line),
                 (frame.shape[1], self.count_line), (0, 255, 0), 2)

        # Прямоугольники вокруг людей
        for person_id, track_data in tracks.items():
            x, y, w, h = track_data['bbox']
            color = (0, 255, 0) if person_id not in self.crossed_ids else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'ID: {person_id}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(frame, f'PeopleCount: {self.people_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, 'Q - exit, R - reset counter', (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def run(self):
        print("Запуск условной логики...")
        print("Можно использовать мышь для изменения позиции линии подсчета")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Обнаружение людей
            people = self.detect_people(frame)

            # Отслеживание
            tracks = self.track_people(people)

            # Проверка пересечения линии
            self.check_crossing(tracks, frame.shape[0])

            # Отрисовка интерфейса
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
                print("Сброс")

        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Итоговое количество людей: {self.people_count}")


class DepartmentCounter(PeopleCounter):
    def __init__(self, department_name, video_source=0, line_position=0.5):
        super().__init__(video_source, line_position)
        self.department_name = department_name
        self.hourly_stats = defaultdict(int)
        self.start_time = time.time()

    def update_hourly_stats(self):
        current_hour = int((time.time() - self.start_time) / 3600)
        self.hourly_stats[current_hour] = self.people_count

    def get_statistics(self):
        stats = {
            'department': self.department_name,
            'total_people': self.people_count,
            'hourly_stats': dict(self.hourly_stats),
            'average_per_hour': self.people_count / max(1, len(self.hourly_stats))
        }
        return stats

if __name__ == "__main__":
    # Для использования камеры
    # counter = StoreDepartmentCounter("Обувь", video_source=0, line_position=0.6)

    # Для чтения видеофайла
    counter = DepartmentCounter("Products", video_source="./example_video_camera.mp4", line_position=0.5)

    try:
        counter.run()
    except KeyboardInterrupt:
        print("\nПрерывание работы...")
    finally:
        stats = counter.get_statistics()
        print(f"\nСтатистика по отделу '{stats['department']}':")
        print(f"Людей в отделе: {stats['total_people']}")
        print(f"Среднее количество людей в час: {stats['average_per_hour']}")
