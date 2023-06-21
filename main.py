import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

from classes.Plate_license_recognition_system import PlateLicenseRecognitionSystem


def process_data():
    args = {}

    data_type = data_type_var.get()
    if data_type:
        args['data_type'] = data_type

    data_path = data_path_var.get()
    if data_path:
        args['data_path'] = data_path

    plate_license = plate_license_var.get().split()
    if plate_license:
        args['plate_license'] = plate_license

    model_path = model_path_var.get()
    if model_path:
        args['model_path'] = model_path

    gpu = gpu_var.get()
    if gpu is not None:
        args['gpu'] = gpu

    match_ratio = match_ratio_var.get()
    if match_ratio is not None:
        args['match_ratio'] = match_ratio

    pls = PlateLicenseRecognitionSystem(**args)
    pls.find_plates(plates_license=plate_license)


def browse_data_path():
    path = filedialog.askopenfilename()
    data_path_var.set(path)


root = tk.Tk()
root.title("Plate License Recognition")
root.geometry("600x400")  # Adjust the window size as needed

data_type_var = tk.StringVar()
data_path_var = tk.StringVar()
plate_license_var = tk.StringVar()
model_path_var = tk.StringVar()
gpu_var = tk.BooleanVar()
match_ratio_var = tk.DoubleVar()

data_type_label = tk.Label(root, text="Data Type:")
data_type_label.pack()
data_type_combobox = ttk.Combobox(root, textvariable=data_type_var, values=["video", "picture"])
data_type_combobox.pack()

data_path_label = tk.Label(root, text="Data Path:")
data_path_label.pack()
data_path_entry = tk.Entry(root, textvariable=data_path_var)
data_path_entry.pack()
data_path_button = tk.Button(root, text="Browse", command=browse_data_path)
data_path_button.pack()

plate_license_label = tk.Label(root, text="Plate License: (optional)")
plate_license_label.pack()
plate_license_entry = tk.Entry(root, textvariable=plate_license_var)
plate_license_entry.pack()

model_path_label = tk.Label(root, text="Model Path: (optional)")
model_path_label.pack()
model_path_entry = tk.Entry(root, textvariable=model_path_var)
model_path_entry.pack()

gpu_checkbutton = tk.Checkbutton(root, text="Use GPU", variable=gpu_var)
gpu_checkbutton.pack()

match_ratio_label = tk.Label(root, text="Match Ratio: (optional)")
match_ratio_label.pack()
match_ratio_entry = tk.Entry(root, textvariable=match_ratio_var)
match_ratio_entry.pack()

process_button = tk.Button(root, text="Process", command=process_data)
process_button.pack()

if __name__ == "__main__":
    root.mainloop()
