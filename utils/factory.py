from models.lwi import LwI  

def get_model(model_name, args):
    name = model_name.lower()
    if name == "lwi":  
        return LwI(args)
    else:
        assert 0  