translate = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
    "dog": "cane",
    "horse": "cavallo",
    "elephant": "elefante",
    "butterfly": "farfalla",
    "chicken": "gallina",
    "cat": "gatto",
    "cow": "mucca",
    "sheep": "pecora",
    "spider": "ragno",
    "squirrel": "scoiattolo"
}


def translate_label(label_es: str) -> str:
    """Renvoie le label traduit si présent, sinon label_es."""
    return translate.get(label_es, label_es)
