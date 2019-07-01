def direct_deleting(root, window):  # Удаление домов из БД напрямую из таблицы
    items = window.table.selection()
    for i in items:
        window.table.delete(i)
        root.dataset.drop((int(i[1:], 16) - 1), inplace=True)
    x = window.table.get_children()
    i = 0
    for item in x:
        window.table.item(item, text=str(i))
        i += 1
