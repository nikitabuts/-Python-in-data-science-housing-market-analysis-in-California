import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from writing_file_window import new_csv


class StaticWindow(tk.Toplevel):  # Окно вывода статистического отчета
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.geometry('420x150')
        self.resizable(True, True)

        self.mass = []
        self.dataset = 0
        self.parameter_label = tk.Label(self, width=30, text='Укажите параметры выборки', anchor='w')
        self.parameter_entry = tk.Listbox(self, width=30, height=5, selectmode=tk.EXTENDED)
        self.scroll = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.parameter_entry.yview)
        self.button = tk.Button(self, text='Найти', command=self.report)

        self.new_table = ttk.Treeview(self, height=13, selectmode=tk.EXTENDED)
        self.save_button = tk.Button(self, command=self.save, text='Сохранить данные')
        self.reset_button = tk.Button(self, command=self.reset, text='Новая выборка')

        for i in range(9):
            self.parameter_entry.insert(tk.END, str(self.master.dataset.columns[i]))
        self.parameter_label.grid(row=0, column=0)
        self.parameter_entry.grid(row=0, column=1)
        self.scroll.grid(row=0, column=2, sticky='w', ipady=20)

        self.button.grid(row=1, column=0, columnspan=3, ipadx='150')

        self.grab_set()
        self.focus_set()

    def report(self):  # Функция вывода статистического отчета по выбранным атрибутам
        if len(self.parameter_entry.curselection()) == 0:
            messagebox.showinfo("Ошибка!", "Не указаны параметры!")
        else:
            self.title('Отчёт')
            self.geometry(str(200*len(self.parameter_entry.curselection())) + 'x400')
            self.parameter_label.grid_remove()
            self.parameter_entry.grid_remove()
            self.scroll.grid_remove()
            self.button.grid_remove()

            rows = ['count', 'mean', 'std', '25%', '50%', '75%', 'max']
            self.new_dataset = self.describing(self.master.dataset)
            self.new_table.column('#0', width=100, stretch=tk.YES)
            self.new_table["columns"] = self.mass
            self.new_table.heading('#0', text='')
            self.new_table.column('#0', width=100, stretch=tk.YES)
            for i in range(len(self.mass)):
                self.new_table.heading(str(self.mass[i]), text=self.mass[i])
                self.new_table.column(self.mass[i], width=100, stretch=tk.YES)
            matrix = self.new_dataset.values

            for i in range(7):
                cells = []
                for j in range(len(self.mass)):
                    try:
                        cells.append(matrix[i][j])
                    except IndexError:
                        break
                tuple(cells)
                self.new_table.insert('', 'end', text=rows[i], values=cells)

            self.new_table.grid(row=0, column=0)
            self.save_button.grid(row=2, column=0, ipadx=100)
            self.reset_button.grid(row=3, column=0, ipadx=100)

    def reset(self):  # Функция для вывода другого отчета по новым атрибутам
        self.geometry('400x200')
        self.new_table.delete(*self.new_table.get_children())
        self.new_table.grid_remove()
        self.save_button.grid_remove()
        self.reset_button.grid_remove()

        self.parameter_label.grid()
        self.parameter_entry.grid()
        self.scroll.grid()
        self.button.grid()

    def save(self):  # Функция сохранения отчета в новый файл
        new_csv(self.master, self.new_dataset, '')

    def describing(self, dataset):
        columns = dataset.columns
        self.mass = [columns[self.parameter_entry.curselection()[i]]
                     for i in range(len(self.parameter_entry.curselection()))]

        new_dataset = dataset[self.mass]
        variable = new_dataset.describe()
        return variable
