class GUIView:
def __init__(self, controller):
self.controller = controller
self.root = tk.Tk()
self.root.title("Login GUI")


tk.Label(self.root, text="Correo:").pack()
self.email = tk.Entry(self.root)
self.email.pack()


tk.Label(self.root, text="Password:").pack()
self.password = tk.Entry(self.root, show="*")
self.password.pack()


tk.Button(self.root, text="Login", command=self.iniciar).pack(pady=5)


self.root.mainloop()


def iniciar(self):
correo = self.email.get()
password = self.password.get()
resultado = self.controller.login(correo, password)
messagebox.showinfo("Resultado", resultado)
