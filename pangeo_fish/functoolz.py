def lookup(mapping, key, message="unknown key: {key}"):
    sentinel = object()
    value = mapping.get(key, sentinel)
    if value is sentinel:
        raise ValueError(message.format(key=key))

    return value


class Pipeline:
    def __init__(self, data):
        self.data = data

    def pipe(self, f, *args, **kwargs):
        result = f(self.data, *args, **kwargs)
        return type(self)(result)
