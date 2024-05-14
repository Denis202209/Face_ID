import tkinter as tk
from PIL import ImageTk, Image
import os

class ImageNavigator:
    def __init__(self, master):
        self.master = master
        self.master.title("Intruders")
        self.image_dir = "intruders/"
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".jpg")])
        self.current_index = 0

        self.image_name_label = tk.Label(master, text=self.image_files[self.current_index])
        self.image_name_label.pack()

        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.show_image(self.current_index)

        prev_button = tk.Button(master, text="Previous", command=self.show_prev_image)
        prev_button.pack(side=tk.LEFT)

        next_button = tk.Button(master, text="Next", command=self.show_next_image)
        next_button.pack(side=tk.LEFT)

        remove_button = tk.Button(master, text="Remove", command=self.remove_image)
        remove_button.pack(side=tk.LEFT)

    def show_image(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        image = Image.open(image_path)
        new_width = int(image.width * 0.5)
        new_height = int(image.height * 0.5)
        resized_image = image.resize((new_width, new_height))
        photo = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=photo)
        self.image_label.image = photo
        self.image_name_label.config(text=self.image_files[index].strip('.jpg'))

    def show_next_image(self):
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.show_image(self.current_index)

    def show_prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self.show_image(self.current_index)

    def remove_image(self):
        image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
        os.remove(image_path)
        del self.image_files[self.current_index]
        if self.current_index >= len(self.image_files):
            self.current_index = 0
        self.show_image(self.current_index)

def show_image_navigator():
    intruders_window = tk.Toplevel()
    navigator = ImageNavigator(intruders_window)

#root = tk.Tk()
#show_button = tk.Button(root, text="Show", command=show_image_navigator)
#show_button.pack()

#root.mainloop()