def parse_hidden_layers(hidden_str):
    try:
        return tuple(int(x.strip()) for x in hidden_str.split(","))
    except:
        return (100,)
