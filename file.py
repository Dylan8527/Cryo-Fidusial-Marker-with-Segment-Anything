import tkinter as tk
from tkinter import filedialog

def select_image_path():
    root = tk.Tk()
    root.withdraw()
    
    filetypes = [
        ('Image files', '*.png;*.jpg;*.jpeg;*.gif;*.bmp'),
        ('PNG files', '*.png'),
        ('JPEG files', '*.jpg;*.jpeg'),
        ('GIF files', '*.gif'),
        ('BMP files', '*.bmp'),
        ('All files', '*.*')
    ]

    file_path = filedialog.askopenfilename(title='选择图片文件',
                                           filetypes=filetypes)

    assert file_path != '', 'No file selected'

    return file_path