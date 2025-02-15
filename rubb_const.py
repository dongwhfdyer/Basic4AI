class _const:
    class ConstError(TypeError): pass
    def __setattr__(self, name, value):
        if self.__dict__.get(name):
            raise self.ConstError("Can't rebind const(%s)" % name)
        self.__dict__[name] = value
    def __delattr__(self, name):
        if self.__dict__.get(name):
            raise self.ConstError("Can't unbind const(%s)" % name)
        del self.__dict__[name]