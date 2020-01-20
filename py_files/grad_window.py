import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tkinter import filedialog
import tkinter as tk
import os


class GradationWindow(tk.Toplevel):  # Окно вывода диаграммы рассеивания в приложении
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.new = self.master.dataset.copy()
        self.title("Диаграмма рассеивания")
        self.im = tk.PhotoImage
        self.label = tk.Label(self)
        self.image = None
        self.california_img = mpimg.imread(filedialog.askopenfilename(defaultextension=".png"))

        self.parameter_label = tk.Label(self, width=30, text='Укажите параметры поиска', anchor='w')
        self.parameter_entry = tk.Listbox(self, width=30, height=8, selectmode=tk.SINGLE)
        for i in range(2, 10):
            self.parameter_entry.insert(tk.END, str(self.master.dataset.columns[i]))

        self.button = tk.Button(self, text='Найти', command=self.grad) # Сохранение диаграммы в новый файл
        self.parameter_entry.grid(row=0, column=1)
        self.button.grid(row=1, column=1)

        self.grad()

        self.save_button = tk.Button(self, text='Сохранить отчёт', command=self.saving)
        self.save_button.grid(row=1, column=0)

        self.grab_set()
        self.focus_set()

    def grad(self):  # Функция создания и отображения диаграммы рассеивания
        self.new.ocean_proximity, value = self.new.ocean_proximity.factorize()
        if len(self.parameter_entry.curselection()) == 0:
            string = 'ocean_proximity'
        else:
            string = self.master.dataset.columns[self.parameter_entry.curselection()[0] + 2]

        self.image = self.new.plot.scatter(x='longitude', y='latitude', s=self.new['population'] / 100,
                                            label='Population', alpha=0.8, c=string, colormap='jet', figsize=(10, 5))
        plt.imshow(self.california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5, cmap=plt.get_cmap('jet'))
        plt.ylabel("Latitude", fontsize=14)
        plt.xlabel("Longitude", fontsize=14)
        plt.title("Распределение классов")
        self.image.figure.savefig('pic.png')
        self.im = tk.PhotoImage(file='pic.png')
        self.label.config(image=self.im)
        self.label.grid(row=0, column=0)
        os.remove('pic.png')

    def saving(self):  # Функция сохранения диаграммы в новый файл в формате png
        file = filedialog.asksaveasfilename(defaultextension=".png")
        if file:
            self.image.figure.savefig(file)
