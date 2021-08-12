class DotConfig(dict):

    def __init__(self, input_dict):
        super(DotConfig, self).__init__(input_dict)

        for k, v in input_dict.items():
            if isinstance(v, dict):
                self[k] = DotConfig(v)
            else:
                self[k] = v

    def update(self, args_dict):
        for k, v in args_dict.items():
            self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)