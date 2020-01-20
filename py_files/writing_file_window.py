from tkinter import filedialog
from tkinter import messagebox


def new_csv(root, dataset, string):
    if string != root.filename:
        string = filedialog.asksaveasfilename(defaultextension=".csv")
        if len(string) == 0:
            messagebox.showinfo("Ошибка!", "Не указаны строки под удаление")
        else:
            with open(string, 'w'):
                pass
            dataset.to_csv(string, encoding='utf-8', index=False)
    else:
        with open(string, 'w'):
            pass
        dataset.to_csv(string, encoding='utf-8', index=False)
