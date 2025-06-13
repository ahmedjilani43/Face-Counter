import time
import os
import cv2
import numpy as np
import multiprocessing
from multiprocessing import Pool, Array
from threading import Thread, Semaphore, Condition
from queue import Queue
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

HAAR_CASCADE_PATH = "C:/Users/ahmed/AppData/Local/Programs/Python/Python313/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"

shared_face_counts = None

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Failed to load image: {image_path}")
        return None
    return img

def detect_faces(image_path, shared_array=None, index=None):
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty():
        raise FileNotFoundError(f"Failed to load Haar Cascade file: {HAAR_CASCADE_PATH}")

    img = load_and_preprocess_image(image_path)
    if img is None:
        return "Vide", 0
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    num_faces = len(faces) if isinstance(faces, np.ndarray) else 0
    label = "Présence humaine" if num_faces > 0 else "Vide"
    if shared_array is not None and index is not None:
        shared_array[index] = num_faces
    return label, num_faces

test_images_dir = "C:/Users/ahmed/Desktop/face_detector/Face-Counter/face_images"
if not os.path.exists(test_images_dir):
    raise FileNotFoundError(f"Directory not found: {test_images_dir}. Please run utils.py to download images.")

test_image_paths = [os.path.join(test_images_dir, fname) for fname in os.listdir(test_images_dir) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not test_image_paths:
    print("Warning: No images found in the directory. Please run utils.py to download images.")

detection_results = []
performance_metrics = []
execution_orders = []


def monothread_face_detection():
    print("Starting Monothread Implementation...")
    start_time = time.time()

    global detection_results
    detection_results = []
    order = []

    for i, image_path in enumerate(test_image_paths, 1):
        label, num_faces = detect_faces(image_path)
        print(f"Image {i} ({os.path.basename(image_path)}): {label} (Faces: {num_faces})")
        detection_results.append((image_path, label, num_faces))
        order.append(f"Image {i}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Monothread Execution Time: {execution_time:.2f} seconds")
    performance_metrics.append({
        "method": "Monothread",
        "execution_time": execution_time,
        "num_images": len(test_image_paths),
        "avg_time_per_image": execution_time / len(test_image_paths) if test_image_paths else 0
    })
    execution_orders.append(("Monothread", order))


def worker_process(args):
    image_path, image_id = args
    label, num_faces = detect_faces(image_path, shared_face_counts, image_id - 1)
    return (image_id, label, num_faces, f"Image {image_id}")

def multiprocess_face_detection():
    global shared_face_counts
    shared_face_counts = Array('i', [0] * len(test_image_paths))
    print("Starting Multiprocess Implementation...")
    start_time = time.time()

    tasks = [(image_path, i) for i, image_path in enumerate(test_image_paths, 1)]

    with Pool(processes=10) as pool:
        results = pool.map(worker_process, tasks)

    results.sort(key=lambda x: x[0])
    order = [result[3] for result in results]
    for image_id, label, num_faces, _ in results:
        print(f"Image {image_id} ({os.path.basename(test_image_paths[image_id-1])}): {label} (Faces: {num_faces})")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Multiprocess Execution Time: {execution_time:.2f} seconds")
    performance_metrics.append({
        "method": "Multiprocess",
        "execution_time": execution_time,
        "num_images": len(test_image_paths),
        "avg_time_per_image": execution_time / len(test_image_paths) if test_image_paths else 0
    })
    execution_orders.append(("Multiprocess", order))


def worker_thread(image_path, image_id, result_queue):
    label, num_faces = detect_faces(image_path)
    result_queue.put((image_id, label, num_faces, f"Image {image_id}"))

def multithread_face_detection():
    print("Starting Multithread Implementation...")
    start_time = time.time()

    threads = []
    result_queue = Queue()
    num_threads = 10
    tasks = [(image_path, i) for i, image_path in enumerate(test_image_paths, 1)]
    tasks_per_thread = (len(tasks) + num_threads - 1) // num_threads

    def thread_worker(thread_id):
        start_idx = thread_id * tasks_per_thread
        end_idx = min((thread_id + 1) * tasks_per_thread, len(tasks))
        for idx in range(start_idx, end_idx):
            image_path, image_id = tasks[idx]
            worker_thread(image_path, image_id, result_queue)

    for thread_id in range(num_threads):
        t = Thread(target=thread_worker, args=(thread_id,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    results.sort(key=lambda x: x[0])
    order = [result[3] for result in results]
    for image_id, label, num_faces, _ in results:
        print(f"Image {image_id} ({os.path.basename(test_image_paths[image_id-1])}): {label} (Faces: {num_faces})")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Multithread Execution Time: {execution_time:.2f} seconds")
    performance_metrics.append({
        "method": "Multithread",
        "execution_time": execution_time,
        "num_images": len(test_image_paths),
        "avg_time_per_image": execution_time / len(test_image_paths) if test_image_paths else 0
    })
    execution_orders.append(("Multithread", order))


def worker_thread_with_semaphore(image_path, image_id, result_queue, print_semaphore):
    label, num_faces = detect_faces(image_path)
    result_queue.put((image_id, label, num_faces, f"Image {image_id}"))
    with print_semaphore:
        print(f"Image {image_id} ({os.path.basename(image_path)}): {label} (Faces: {num_faces}) (synchronized)")

def multithread_with_semaphore():
    print("Starting Multithread Implementation with Semaphore...")
    start_time = time.time()

    threads = []
    result_queue = Queue()
    print_semaphore = Semaphore(1)
    num_threads = 10
    tasks = [(image_path, i) for i, image_path in enumerate(test_image_paths, 1)]
    tasks_per_thread = (len(tasks) + num_threads - 1) // num_threads

    def thread_worker(thread_id):
        start_idx = thread_id * tasks_per_thread
        end_idx = min((thread_id + 1) * tasks_per_thread, len(tasks))
        for idx in range(start_idx, end_idx):
            image_path, image_id = tasks[idx]
            worker_thread_with_semaphore(image_path, image_id, result_queue, print_semaphore)

    for thread_id in range(num_threads):
        t = Thread(target=thread_worker, args=(thread_id,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    results.sort(key=lambda x: x[0])
    order = [result[3] for result in results]
    for image_id, label, num_faces, _ in results:
        pass  

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Multithread with Semaphore Execution Time: {execution_time:.2f} seconds")
    performance_metrics.append({
        "method": "Semaphore",
        "execution_time": execution_time,
        "num_images": len(test_image_paths),
        "avg_time_per_image": execution_time / len(test_image_paths) if test_image_paths else 0
    })
    execution_orders.append(("Semaphore", order))

def producer(image_path, image_id, shared_queue, condition):
    label, num_faces = detect_faces(image_path)
    with condition:
        shared_queue.put((image_id, image_path, label, num_faces, f"Image {image_id}"))
        condition.notify()

def consumer(shared_queue, condition, num_producers):
    processed = 0
    order = []
    while processed < num_producers:
        with condition:
            while shared_queue.empty():
                condition.wait()
            image_id, image_path, label, num_faces, img_name = shared_queue.get()
            print(f"Consumer: Image {image_id} ({os.path.basename(image_path)}): {label} (Faces: {num_faces})")
            order.append(img_name)
            processed += 1
    return order

def producer_consumer_face_detection():
    print("Starting Producer/Consumer Implementation...")
    start_time = time.time()

    shared_queue = Queue()
    condition = Condition()
    num_producers = len(test_image_paths)
    num_threads = 10
    tasks = [(image_path, i) for i, image_path in enumerate(test_image_paths, 1)]
    tasks_per_thread = (len(tasks) + num_threads - 1) // num_threads

    consumer_thread = Thread(target=lambda: execution_orders.append(("Producer/Consumer", consumer(shared_queue, condition, num_producers))))
    consumer_thread.start()

    producers = []
    def producer_worker(thread_id):
        start_idx = thread_id * tasks_per_thread
        end_idx = min((thread_id + 1) * tasks_per_thread, len(tasks))
        for idx in range(start_idx, end_idx):
            image_path, image_id = tasks[idx]
            producer(image_path, image_id, shared_queue, condition)

    for thread_id in range(num_threads):
        p = Thread(target=producer_worker, args=(thread_id,))
        producers.append(p)
        p.start()

    for p in producers:
        p.join()

    consumer_thread.join()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Producer/Consumer Execution Time: {execution_time:.2f} seconds")
    performance_metrics.append({
        "method": "Producer/Consumer",
        "execution_time": execution_time,
        "num_images": len(test_image_paths),
        "avg_time_per_image": execution_time / len(test_image_paths) if test_image_paths else 0
    })

def display_results_gui():
    root = tk.Tk()
    root.title("Face Detection Results")
    root.configure(bg="#f0f0f0")

    notebook = ttk.Notebook(root)
    notebook.pack(pady=10, expand=True, fill="both")

    # Tab 1: Detection Results
    results_tab = ttk.Frame(notebook)
    notebook.add(results_tab, text="Detection Results")

    results_frame = ttk.Frame(results_tab, padding="10")
    results_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    for idx, (image_path, label, num_faces) in enumerate(detection_results):
        try:
            img = Image.open(image_path)
            img = img.resize((150, 150), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Warning: Failed to load image for display: {image_path} ({e})")
            continue

        row = idx // 3
        col = idx % 3

        img_frame = ttk.Frame(results_frame, padding="5")
        img_frame.grid(row=row, column=col, padx=10, pady=10)

        img_label = ttk.Label(img_frame, image=photo)
        img_label.image = photo
        img_label.grid(row=0, column=0)

        label_text = f"{label} (Faces: {num_faces})"
        label_color = "green" if label == "Présence humaine" else "red"
        pred_label = ttk.Label(img_frame, text=label_text, foreground=label_color, font=("Arial", 12, "bold"))
        pred_label.grid(row=1, column=0, pady=5)

        filename_label = ttk.Label(img_frame, text=os.path.basename(image_path), font=("Arial", 10))
        filename_label.grid(row=2, column=0)

    performance_tab = ttk.Frame(notebook)
    notebook.add(performance_tab, text="Performance Metrics")

    performance_frame = ttk.Frame(performance_tab, padding="10")
    performance_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    columns = ("Method", "Execution Time (s)", "Images Processed", "Avg Time per Image (s)")
    tree = ttk.Treeview(performance_frame, columns=columns, show="headings")
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=150, anchor="center")
    tree.grid(row=0, column=0, pady=10)

    for metric in performance_metrics:
        tree.insert("", "end", values=(
            metric["method"],
            f"{metric['execution_time']:.2f}",
            metric["num_images"],
            f"{metric['avg_time_per_image']:.3f}"
        ))

    fig, ax = plt.subplots(figsize=(8, 4))
    methods = [metric["method"] for metric in performance_metrics]
    times = [metric["execution_time"] for metric in performance_metrics]
    colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99", "#FF99CC"]
    ax.bar(methods, times, color=colors)
    ax.set_title("Execution Time Comparison", fontsize=14, pad=15)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=performance_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=0, pady=10)

    fastest = min(performance_metrics, key=lambda x: x["execution_time"]) if performance_metrics else {"method": "N/A", "execution_time": 0}
    slowest = max(performance_metrics, key=lambda x: x["execution_time"]) if performance_metrics else {"method": "N/A", "execution_time": 0}
    summary_text = (
        f"Fastest Method: {fastest['method']} ({fastest['execution_time']:.2f} seconds)\n"
        f"Slowest Method: {slowest['method']} ({slowest['execution_time']:.2f} seconds)\n\n"
        "Observations:\n"
        "- Multiprocess uses 10 processes with a shared memory array for face counts, leveraging parallel CPU usage.\n"
        "- Multithread, Semaphore, and Producer/Consumer use 10 threads, suitable for I/O-bound tasks but now thread-safe with individual classifiers.\n"
        "- Monothread is simple and sequential, potentially faster for small datasets due to no overhead.\n"
        "- The shared memory array in multiprocess ensures inter-process communication for face counts."
    )
    summary_label = ttk.Label(performance_frame, text=summary_text, font=("Arial", 10), justify="left")
    summary_label.grid(row=2, column=0, pady=10)

    order_text = "Execution Order:\n"
    for method, order in execution_orders:
        order_text += f"- {method}: {', '.join(order)}\n"
    order_label = ttk.Label(performance_frame, text=order_text, font=("Arial", 10), justify="left")
    order_label.grid(row=3, column=0, pady=10)

    root.mainloop()
    SystemExit

if __name__ == "__main__":
    print("=== Running All Implementations ===")
    monothread_face_detection()
    print("\n")
    multiprocess_face_detection()
    print("\n")
    multithread_face_detection()
    print("\n")
    multithread_with_semaphore()
    print("\n")
    producer_consumer_face_detection()
    print("\n")
    print("Displaying Results in GUI...")
    display_results_gui()