from pymoo.core.callback import Callback

class Callback(Callback):
    def __init__(self):
        super().__init__()
        self.data["po_paths"] = []
        self.data["po_f_values"] = []
        self.data["all_paths"] = []
        self.data["all_f_values"] = []


    def notify(self, algorithm):
        #print(algorithm.pop)
        self.data["all_paths"].append(algorithm.pop.get("X"))
        self.data["all_f_values"].append(algorithm.pop.get("F"))
        self.data["po_paths"].append(algorithm.opt.get("X"))
        self.data["po_f_values"].append(algorithm.opt.get("F"))