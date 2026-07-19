from __future__ import annotations

import tkinter as tk


def set_var_if_changed(var: tk.StringVar, value: object) -> bool:
    text = str(value)
    if var.get() == text:
        return False
    var.set(text)
    return True
