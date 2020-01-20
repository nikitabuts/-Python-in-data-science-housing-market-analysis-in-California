import tkinter as tk
import numpy as np
import pandas as pd
from tkinter import messagebox

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


class AddingWindow(tk.Toplevel):
    def __init__(self, master, window):
        super().__init__()
        self.master = master
        self.window = window
        self.classes = self.master.dataset['ocean_proximity'].unique()
        self.flag_prediction1 = False
        self.flag_prediction2 = False

        self.longitude_label = tk.Label(self, width=30, text='Введите longitude', anchor='w')
        self.longitude_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 0]),
                                          to=np.max(self.master.dataset.iloc[:, 0]))

        self.latitude_label = tk.Label(self, width=30, text='Введите latitude', anchor='w')
        self.latitude_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 1]),
                                         to=np.max(self.master.dataset.iloc[:, 1]))

        self.housing_median_age_label = tk.Label(self, width=30, text='Введите housing_median_age', anchor='w')
        self.housing_median_age_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 2]),
                                                   to=np.max(self.master.dataset.iloc[:, 2]))

        self.total_rooms_label = tk.Label(self, width=30, text='Введите total_rooms', anchor='w')
        self.total_rooms_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 3]),
                                            to=np.max(self.master.dataset.iloc[:, 3]))

        self.total_bedrooms_label = tk.Label(self, width=30, text='Введите total_bedrooms', anchor='w')
        self.total_bedrooms_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 4]),
                                               to=np.max(self.master.dataset.iloc[:, 4]))

        self.population_label = tk.Label(self, width=30, text='Введите population', anchor='w')
        self.population_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 5]),
                                           to=np.max(self.master.dataset.iloc[:, 5]))

        self.households_label = tk.Label(self, width=30, text='Введите households', anchor='w')
        self.households_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 6]),
                                           to=np.max(self.master.dataset.iloc[:, 6]))

        self.median_income_label = tk.Label(self, width=30, text='Введите median_income', anchor='w')
        self.median_income_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 7]),
                                              to=np.max(self.master.dataset.iloc[:, 7]))

        self.median_house_value_label = tk.Label(self, width=30, text='Введите median_house_value', anchor='w')
        self.median_house_value_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 9]),
                                                   to=np.max(self.master.dataset.iloc[:, 9]))

        self.ocean_proximity_label = tk.Label(self, width=30, text='Введите ocean_proximity', anchor='w')
        self.ocean_proximity_entry = tk.Listbox(self, width=30, height=6, selectmode=tk.SINGLE)

        self.price_predict_label = tk.Label(self, width=30, text='Хотите ли вы, чтобы цена квартиры\n'
                                                                 ' прогнозировалась автоматически?(y/n)', anchor='w')
        self.price_predict_entry = tk.Spinbox(self, width=30)

        self.class_predict_label = tk.Label(self, width=30, text='Хотите ли вы, чтобы зона квартиры\n'
                                                                 ' прогнозировалась автоматически?(y/n)', anchor='w')
        self.class_predict_entry = tk.Spinbox(self)

        self.Button = tk.Button(self, text='Добавить', command=self.apply)

        self.init_child()

    def init_child(self):
        self.title('Добавить запись')
        self.geometry('460x385+100+100')
        self.resizable(False, False)

        self.longitude_label.grid(row=0, column=0)
        self.longitude_entry.grid(row=0, column=1)

        self.latitude_label.grid(row=1, column=0)
        self.latitude_entry.grid(row=1, column=1)

        self.housing_median_age_label.grid(row=2, column=0)
        self.housing_median_age_entry.grid(row=2, column=1)

        self.total_rooms_label.grid(row=3, column=0)
        self.total_rooms_entry.grid(row=3, column=1)

        self.total_bedrooms_label.grid(row=4, column=0)
        self.total_bedrooms_entry.grid(row=4, column=1)

        self.population_label.grid(row=5, column=0)
        self.population_entry.grid(row=5, column=1)

        self.households_label.grid(row=6, column=0)
        self.households_entry.grid(row=6, column=1)

        self.median_income_label.grid(row=7, column=0)
        self.median_income_entry.grid(row=7, column=1)

        self.median_house_value_label.grid(row=8, column=0)
        self.median_house_value_entry.grid(row=8, column=1)

        self.ocean_proximity_label.grid(row=9, column=0)
        for i in range(len(self.classes)):
            self.ocean_proximity_entry.insert(tk.END, self.classes[i])
        self.ocean_proximity_entry.grid(row=9, column=1)

        self.price_predict_label.grid(row=10, column=0, sticky='w')
        self.price_predict_entry.grid(row=10, column=1, sticky='w')

        self.class_predict_label.grid(row=11, column=0, sticky='w')
        self.class_predict_entry.grid(row=11, column=1,sticky='w')

        self.Button.grid(row=12, column=0, columnspan=2, ipadx=200)

        self.grab_set()
        self.focus_set()

    def apply(self):
        flag = True
        if self.price_predict_entry.get() == 'y':
            self.flag_prediction1 = True
        if self.class_predict_entry.get() == 'y':
            self.flag_prediction2 = True
        if len(self.ocean_proximity_entry.curselection()) == 0:
            flag = False
            messagebox.showinfo("Ошибка!", "Не указан параметр ocean_proximity_entry!")
        else:
            dictionary = dict(zip(np.arange(0, 10).tolist(),
                            [self.longitude_entry.get(),
                                 self.latitude_entry.get(),
                                 self.housing_median_age_entry.get(),
                                 self.total_bedrooms_entry.get(),
                                 self.total_bedrooms_entry.get(),
                                 self.population_entry.get(),
                                 self.households_entry.get(),
                                 self.median_income_entry.get(),
                                 self.classes[self.ocean_proximity_entry.curselection()[0]],
                                 self.median_house_value_entry.get()]))

            for i in range(0, 10):
                try:
                    if i != 8:
                        variable = float(dictionary[i])
                    else:
                        variable = dictionary[i]
                except ValueError:
                    flag = False
                    messagebox.showinfo("Ошибка!", "Неверный тип переменной " + str(self.master.dataset.columns[i]) + "!")

            if len(self.ocean_proximity_entry.curselection()) == 0:
                flag = False
                messagebox.showinfo("Ошибка!", "Не указан параметр ocean_proximity_entry!")

            if flag:
                data_frame = self.appending(dictionary)
                self.master.dataset = self.master.dataset.append(data_frame, ignore_index=True)
                matrix = data_frame.values
                cells = []
                for j in range(10):
                    try:
                        cells.append(matrix[0][j])
                    except IndexError:
                        break
                tuple(cells)
                self.window.table.insert('', 'end', text=str((len(self.master.dataset.longitude) - 1)), values=cells)

    def appending(self, dict):
        ocean_proximity_array = self.master.dataset.ocean_proximity.unique()
        auxiliary_array = np.array([])
        columns_of_dataset = self.master.dataset.columns

        for iteration in np.arange(self.master.dataset.shape[1]):
            if columns_of_dataset[iteration] == 'median_house_value' and self.flag_prediction1:
                # проверка на то, что найден именно столбец median_house_value
                x = self.master.dataset.drop(['median_house_value', 'ocean_proximity'], axis=1)
                y = self.master.dataset.median_house_value
                auxiliary_sub_array = auxiliary_array[:-1].astype(float)

                regression = LinearRegression()
                regression.fit(x, y)
                variable = int(regression.predict(auxiliary_sub_array))

            elif columns_of_dataset[iteration] == 'ocean_proximity' and self.flag_prediction2:
                auxiliary_sub_array = [auxiliary_array[0:2]]
                classificator = MultiClassification(self.master.dataset)
                classificator.fit()
                variable = classificator.predict(auxiliary_sub_array)
            else:
                variable = dict[iteration]

            auxiliary_array = np.append(auxiliary_array, variable)

        df = pd.DataFrame(columns=self.master.dataset.columns)

        df.loc[len(self.master.dataset)] = [float(auxiliary_array[0]), float(auxiliary_array[1]),
                                            float(auxiliary_array[2]), float(auxiliary_array[3]),
                                            float(auxiliary_array[4]), float(auxiliary_array[5]),
                                            float(auxiliary_array[6]), float(auxiliary_array[7]),
                                            (auxiliary_array[8]), float(auxiliary_array[9])]

        return df


class MultiClassification:
    def __init__(self, dataset):
        self.__dataset = dataset

    def __class_balancing(self):
        helped_dataset = self.__dataset.copy()
        helped_dataset.ocean_proximity, _ = helped_dataset.ocean_proximity.factorize()
        series = pd.Series(helped_dataset.ocean_proximity)
        maximum = series.value_counts().max()

        x = pd.DataFrame(columns=['longitude', 'latitude', 'ocean_proximity']).values

        for i in np.arange(len(series.unique())):
            helped_2 = series.unique()[i]
            array = helped_dataset[helped_dataset['ocean_proximity'] == helped_2].loc[:,
                    ['longitude', 'latitude', 'ocean_proximity']].values.reshape(-1, 3)
            help_3 = array.copy()
            while len(array) < maximum:
                if maximum - len(array) >= len(help_3):
                    array = np.append(array, help_3, axis=0)
                else:
                    array = np.append(array, help_3[0: maximum - len(help_3)], axis=0)
            x = np.append(x, array, axis=0)

        x = pd.DataFrame(x, columns=['longitude', 'latitude', 'ocean_proximity'])
        return x

    def fit(self):
        balanced_dataset = self.__class_balancing()

        self.y = pd.DataFrame(balanced_dataset).iloc[:, 2].values.reshape((len(balanced_dataset),)).astype(int)
        self.x = balanced_dataset.drop('ocean_proximity', axis=1).values.astype(float)

        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

        for train_index, test_index in skf.split(self.x, self.y):
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

        self.clf = RandomForestClassifier(n_estimators=500, max_depth=20)
        self.clf.fit(x_train, y_train)

    def predict(self, x_test):
        y_pred = self.clf.predict(x_test)
        labels_unique = pd.Series(self.y).unique()
        preds = self.clf.predict(x_test)
        for i in range(len(labels_unique)):
            if preds == labels_unique[i]:
                preds = self.__dataset.ocean_proximity.unique()[i]
        return preds


class LinearRegression:
    def __init__(self):
        self.weights = np.array([])

    def fit(self, x, y):
        x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
        self.weights = np.linalg.inv(x.T @ x) @ x.T @ y.values

    def predict(self, x):
        bias = self.weights[0]
        self.weights = self.weights[1: len(self.weights)]
        predictions = bias + x @ self.weights.T
        return predictions

    @staticmethod
    def score(y_test, y_pred):
        absolute_error = np.abs(y_test - y_pred)
        epsilons = list(map(float, input().split()))
        scores = np.array([])
        for i in np.arange(len(epsilons)):
            summary = 0
            for j in np.arange(len(absolute_error)):
                if absolute_error[j] <= epsilons[i]:
                    summary += 1
            score = summary / len(absolute_error)
            scores = np.append(scores, score)
        scores_df = pd.DataFrame({"Epsilon": epsilons, "Score": scores})
        return scores_df
