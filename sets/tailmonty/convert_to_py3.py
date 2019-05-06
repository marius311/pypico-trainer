import pickle

indatafile  = "pico3_tailmonty_v35.dat"
outdatafile = "pico3_tailmonty_v35_py3.dat"

with open(indatafile,"rb") as f:
    d = pickle.load(f,encoding="latin1")

with open(indatafile,"rb") as f:
    db = pickle.load(f,encoding="bytes")

d["code"] = open("tailmonty.py").read()
d["version"] = "4.0.0"
d["module_name"]
d["pico"] = db[b"pico"]

with open(outdatafile,'wb') as f: pickle.dump(d,f)

#%% Test
 
with open(outdatafile,"rb") as f: data = pickle.load(f)
assert type(data["pico"]) == bytes

import pypico
pico = pypico.load_pico(outdatafile)
print(pico.get(**pico.example_inputs()))

print("Succesfully wrote '%s'"%outdatafile)
