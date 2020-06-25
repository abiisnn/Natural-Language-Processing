from gensim.summarization.summarizer import summarize

text = """
Los elefantes o elefántidos (Elephantidae) 
son una familia de mamíferos placentarios del orden Proboscidea. 
Antiguamente se clasificaban, junto con otros mamíferos de piel gruesa, 
en el orden, ahora inválido, de los paquidermos (Pachydermata). 
Existen hoy en día tres especies y diversas subespecies. 
Entre los géneros extintos de esta familia destacan los mamuts.
Elefante salvaje indio del bosque de Marayoor, Munnar, Kerala
Cría de elefante africano de sabana (Loxodonta africana), 
parque nacional Kruger, Sudáfrica
Los elefantes son los animales terrestres más grandes que 
existen en la actualidad. El periodo de gestación es de veintidós meses, 
el más largo en cualquier animal terrestre. 
El peso al nacer usualmente es 118 kg. 
Normalmente viven de cincuenta a setenta años, 
pero registros antiguos documentan edades máximas de ochenta y dos años. 
El elefante más grande que se ha cazado, de los que se tiene registro, 
pesó alrededor de 11 000 kg (Angola, 1956), 
alcanzando una altura en la cruz de 3,96 m, un metro más alto que el elefante 
africano promedio. El elefante más pequeño, de alrededor del tamaño de una cría o un 
cerdo grande, es una especie prehistórica que existió en la isla de Creta, 
Elephas creticus, durante el Pleistoceno.
Con un peso de 5 kg, el cerebro del elefante es el más grande de los 
animales terrestres. Se le atribuyen una gran variedad de comportamientos 
asociados a la inteligencia como el duelo, altruismo, adopción, juego, 
uso de herramientas, compasión y autorreconocimiento. 
Los elefantes pueden estar a la par con otras especies inteligentes como los cetáceos
y algunos primates. Las áreas más grandes en su cerebro están encargadas de la audición, 
el gusto y la movilidad.
"""
ans = summarize(text, ratio=0.2, split=True)
print(ans)