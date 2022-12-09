from tkinter import *
from tkinter import ttk as tk
import os

def save_message(path, name):
    with open(r'{path}{name}', 'w', encoding='utf8') as file:
        file.write("/////")


window = Tk()

window.geometry("1000x800")

Frame = tk.Frame(window).grid(row=10, column=10)

website_name = 'GOOGLE'

tk.Label(Frame, text=website_name, relief="flat",
         foreground="green", background="#d0e8fc", padding=10,
         font=("Arial", 14)).grid(column=10, row=10)

your_motto = 'Get better!'

# tk.Label(Frame, text=your_motto, relief="groove",
#          foreground="green", background="#FFCDD2", padding=10,
#          font=("Arial", 14)).grid(column=5, row=5)

button_text = 'Make folder'

# tk.Button(Frame, text=button_text, command=window.destroy,
# padding=10).grid(column=7, row=8)

# tk.Entry(width=30).grid(row=0, column=1, columnspan=3)

window.mainloop()
