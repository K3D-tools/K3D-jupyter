from base64 import b64decode

with open("./log.txt", "r") as f:
    log = f.read().splitlines()

for i in range(len(log)):
    if (
            log[i]
            == "----------------------------- Captured stdout call -----------------------------"
    ):
        name = log[i + 2]
        print(name)
        base64 = log[i + 3][2:-1]

        with open(name + ".png", "wb") as f:
            f.write(b64decode(base64))
